"""Hard-tasks mini-benchmark.

Goal: find tasks where Haiku 4.5 is at its failure boundary so that
configuration differences (MaTTS k=1 vs k=4) actually show up in accuracy.

We use classic reasoning traps + multi-step puzzles. We run each task multiple
times per config to estimate accuracy under sampling temperature.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fleet.agents.agent import Agent
from fleet.core.state import GraphState
from fleet.memory.bank import ReasoningBank
from fleet.memory.embedders.minilm import MiniLMEmbedder
from fleet.memory.matts import matts_run
from fleet.memory.stores.inmemory import InMemoryStore
from fleet.providers.client import FleetLLM

# We reuse the same harness pieces from the main script.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_benchmarks import (  # noqa: E402
    MODEL, BACKEND, INTER_CALL_SLEEP_S, RESULTS_DIR,
    PRICE_INPUT_PER_M, PRICE_OUTPUT_PER_M,
    Task, RunMetrics, verify,
    _collect_tool_calls, _final_text, _count_memory_block_uses,
    _RunnableAgent,
)


HARD_TASKS: list[Task] = [
    Task(
        id="trap_bat_ball",
        category="trap",
        prompt=(
            "A bat and a ball together cost $1.10. The bat costs $1.00 more than "
            "the ball. How much does the ball cost in dollars? "
            "Reply with just the number in dollars (e.g. 0.42)."
        ),
        expected=["0.05"],
    ),
    Task(
        id="trap_widgets",
        category="trap",
        prompt=(
            "If 5 machines take 5 minutes to make 5 widgets, how many minutes "
            "does it take 100 machines to make 100 widgets? "
            "Reply with just the number of minutes."
        ),
        expected=["5"],
    ),
    Task(
        id="trap_snail",
        category="trap",
        prompt=(
            "A snail at the bottom of a 10-meter pole climbs 3 meters up each "
            "day, then slides 2 meters down each night. On what day number does "
            "it first reach the top? Reply with just the day number."
        ),
        expected=["8"],
    ),
    Task(
        id="trap_sheep",
        category="trap",
        prompt=(
            "A farmer has 17 sheep. All but 9 die. How many sheep are left? "
            "Reply with just the number."
        ),
        expected=["9"],
    ),
    Task(
        id="trap_handshakes",
        category="trap",
        prompt=(
            "There are 6 people at a party. Each person shakes hands with every "
            "other person exactly once. How many handshakes occur in total? "
            "Reply with just the number."
        ),
        expected=["15"],
    ),
]


def _cost(in_tok: int, out_tok: int) -> float:
    return (in_tok * PRICE_INPUT_PER_M + out_tok * PRICE_OUTPUT_PER_M) / 1_000_000.0


async def run_single(task: Task, llm: FleetLLM, *, bank: ReasoningBank | None) -> RunMetrics:
    agent = Agent(
        llm=llm, tools=task.tools, max_iters=5,
        memory_bank=bank, memory_k=3 if bank else 0,
    )
    state = GraphState(goal=task.prompt)
    tracker = {"in": 0, "out": 0}
    orig = llm.complete

    async def tracked(messages, tools=None):
        resp = await orig(messages, tools=tools)
        if llm.usage:
            tracker["in"] += int(getattr(llm.usage, "input_tokens", 0) or 0)
            tracker["out"] += int(getattr(llm.usage, "output_tokens", 0) or 0)
        return resp

    llm.complete = tracked  # type: ignore[assignment]
    t0 = time.perf_counter()
    try:
        final = await agent.step(state)
    finally:
        llm.complete = orig  # type: ignore[assignment]
    wall = time.perf_counter() - t0
    tools = _collect_tool_calls(final)
    text = _final_text(final)
    ok, note = verify(task, text, tools)
    return RunMetrics(
        task_id=task.id, config="", success=ok, note=note,
        wall_s=round(wall, 3), input_tokens=tracker["in"], output_tokens=tracker["out"],
        total_tokens=tracker["in"] + tracker["out"],
        cost_usd=round(_cost(tracker["in"], tracker["out"]), 6),
        tool_calls=tools, final_text=text[:300],
        memories_used=_count_memory_block_uses(final) if bank else 0,
    )


async def run_matts(task: Task, llm: FleetLLM, *, k: int, bank: ReasoningBank) -> RunMetrics:
    agent = Agent(llm=llm, tools=task.tools, max_iters=5, memory_bank=bank, memory_k=3)
    runnable = _RunnableAgent(agent)
    tracker = {"in": 0, "out": 0}
    orig = llm.complete

    async def tracked(messages, tools=None):
        resp = await orig(messages, tools=tools)
        if llm.usage:
            tracker["in"] += int(getattr(llm.usage, "input_tokens", 0) or 0)
            tracker["out"] += int(getattr(llm.usage, "output_tokens", 0) or 0)
        return resp

    llm.complete = tracked  # type: ignore[assignment]
    t0 = time.perf_counter()
    try:
        states, distilled = await matts_run(runnable, task.prompt, bank=bank, k=k)
    finally:
        llm.complete = orig  # type: ignore[assignment]
    wall = time.perf_counter() - t0

    # Self-consistency: pick the answer that appears most often across rollouts.
    answers: list[str] = []
    per_state_ok: list[bool] = []
    for st in states:
        text = _final_text(st)
        tools_used = _collect_tool_calls(st)
        ok, _ = verify(task, text, tools_used)
        per_state_ok.append(ok)
        # Extract last number-ish or strip
        from run_benchmarks import _NUMBER_RE
        nums = _NUMBER_RE.findall(text)
        ans = nums[-1] if nums else text.strip().lower()[:30]
        answers.append(ans)

    # majority vote
    counts: dict[str, int] = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    best_ans = max(counts, key=counts.get) if counts else ""
    # Pick the state matching the majority answer (or first OK).
    best_state = states[0]
    best_text = _final_text(best_state)
    best_tools = _collect_tool_calls(best_state)
    for st, a in zip(states, answers):
        if a == best_ans:
            best_state = st
            best_text = _final_text(st)
            best_tools = _collect_tool_calls(st)
            break

    ok, note = verify(task, best_text, best_tools)

    return RunMetrics(
        task_id=task.id, config="", success=ok,
        note=f"vote={best_ans} per_rollout={per_state_ok} {note}",
        wall_s=round(wall, 3),
        input_tokens=tracker["in"], output_tokens=tracker["out"],
        total_tokens=tracker["in"] + tracker["out"],
        cost_usd=round(_cost(tracker["in"], tracker["out"]), 6),
        tool_calls=best_tools, final_text=best_text[:300],
        memories_used=_count_memory_block_uses(best_state),
        memories_distilled=len(distilled),
    )


async def main() -> None:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    # Use temperature 0.7 across the board so k=1 and MaTTS share a sampler.
    llm = FleetLLM(BACKEND, MODEL, max_tokens=512, temperature=0.7, api_key=api_key)
    embedder = MiniLMEmbedder()

    REPEATS = 3  # repeat each task for stochastic accuracy estimates
    all_results: list[RunMetrics] = []

    # --- Config A: baseline k=1 -----------------------------------------
    print("\n=== HARD-A: baseline k=1 (no bank), 3 repeats ===", flush=True)
    for rep in range(REPEATS):
        for task in HARD_TASKS:
            m = await run_single(task, llm, bank=None)
            m.config = f"hard_k1_rep{rep}"
            print(
                f"  rep{rep} [{task.id:18s}] ok={m.success} t={m.wall_s:5.2f}s "
                f"in={m.input_tokens} out={m.output_tokens} note={m.note}",
                flush=True,
            )
            all_results.append(m)
            await asyncio.sleep(INTER_CALL_SLEEP_S)
    await asyncio.sleep(3)

    # --- Config B: MaTTS k=4 majority vote --------------------------------
    print("\n=== HARD-B: MaTTS k=4 majority vote (no bank), 3 repeats ===", flush=True)
    # Empty bank that just provides induction_llm for the contrast distill.
    bank = ReasoningBank(
        store=InMemoryStore(), embedder=embedder,
        induction_llm=llm, judge_llm=llm, async_writeback=False,
    )
    for rep in range(REPEATS):
        for task in HARD_TASKS:
            m = await run_matts(task, llm, k=4, bank=bank)
            m.config = f"hard_k4_rep{rep}"
            print(
                f"  rep{rep} [{task.id:18s}] ok={m.success} t={m.wall_s:5.2f}s "
                f"in={m.input_tokens} out={m.output_tokens} note={m.note}",
                flush=True,
            )
            all_results.append(m)
            await asyncio.sleep(INTER_CALL_SLEEP_S * 3)
    await asyncio.sleep(3)

    # --- aggregate -------------------------------------------------------
    raw_path = RESULTS_DIR / "hard_results.json"
    with raw_path.open("w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nWrote hard raw → {raw_path}", flush=True)

    # Compute per-task accuracy across reps for each config family
    families = {"k1": "hard_k1_rep", "k4_matts": "hard_k4_rep"}
    summary: dict = {}
    for fam, prefix in families.items():
        rows = [r for r in all_results if r.config.startswith(prefix)]
        per_task: dict[str, dict] = {}
        for r in rows:
            d = per_task.setdefault(r.task_id, {"runs": 0, "succ": 0})
            d["runs"] += 1
            d["succ"] += 1 if r.success else 0
        overall_runs = sum(d["runs"] for d in per_task.values())
        overall_succ = sum(d["succ"] for d in per_task.values())
        summary[fam] = {
            "overall_runs": overall_runs,
            "overall_succ": overall_succ,
            "overall_accuracy": round(overall_succ / overall_runs, 3),
            "input_tokens": sum(r.input_tokens for r in rows),
            "output_tokens": sum(r.output_tokens for r in rows),
            "cost_usd": round(sum(r.cost_usd for r in rows), 5),
            "wall_s": round(sum(r.wall_s for r in rows), 2),
            "per_task": {
                tid: {**d, "accuracy": round(d["succ"] / d["runs"], 3)}
                for tid, d in per_task.items()
            },
        }
    sum_path = RESULTS_DIR / "hard_summary.json"
    with sum_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote hard summary → {sum_path}", flush=True)

    print("\n=== HARD SUMMARY ===")
    for fam, s in summary.items():
        print(
            f"  {fam}: acc={s['overall_accuracy']:.2f} "
            f"({s['overall_succ']}/{s['overall_runs']}) "
            f"cost=${s['cost_usd']:.4f} wall={s['wall_s']:.1f}s"
        )
        for tid, d in s["per_task"].items():
            print(f"     {tid:18s} {d['succ']}/{d['runs']}  acc={d['accuracy']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
