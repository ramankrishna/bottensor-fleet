"""Benchmark harness for MaTTS and ReasoningBank features.

Compares:
- Single-pass (k=1) vs MaTTS parallel rollouts (k=2, k=4)
- With vs without ReasoningBank (memory-augmented retrieval)
- Cost, latency, accuracy, tool-use success

Designed to be friendly to a 30K-tokens-per-minute dev-tier key:
- Runs Claude Haiku 3.5 for every LLM call
- Inserts inter-run sleeps to amortize token usage over time
- Keeps per-task max_iters low so each run finishes in a few turns

Outputs:
- benchmark_results/results.json  (raw)
- benchmark_results/summary.json   (aggregate)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fleet.agents.agent import Agent
from fleet.core.state import GraphState
from fleet.memory.bank import ReasoningBank
from fleet.memory.embedders.minilm import MiniLMEmbedder
from fleet.memory.matts import matts_run
from fleet.memory.stores.inmemory import InMemoryStore
from fleet.providers.client import FleetLLM

# --------------------------------------------------------------------------- config

MODEL = "claude-haiku-4-5-20251001"  # smallest model available on this key
BACKEND = "anthropic"
MAX_ITERS = 5
INTER_CALL_SLEEP_S = 1.2  # gentle pacing for 30K TPM rate limit
INTER_CONFIG_SLEEP_S = 4.0
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmark_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Anthropic Haiku 4.5 pricing (USD per 1M tokens) — used for cost estimation.
PRICE_INPUT_PER_M = 1.00
PRICE_OUTPUT_PER_M = 5.00


# --------------------------------------------------------------------------- tasks


@dataclass
class Task:
    id: str
    category: str
    prompt: str
    expected: list[str]  # list of acceptable answer substrings (case-insensitive)
    tools: list[str] = field(default_factory=list)
    tool_required: str | None = None  # if set, agent must call this tool to succeed


# Reasoning / math / code / tool-use tasks. Each has a verifier-friendly answer.
TASKS: list[Task] = [
    Task(
        id="math_train",
        category="math",
        prompt=(
            "A train travels at 60 mph for 2.5 hours. "
            "How many miles does it cover? Reply with just the number."
        ),
        expected=["150"],
    ),
    Task(
        id="math_even_sum",
        category="math",
        prompt=(
            "What is the sum of the first 10 positive even numbers "
            "(2, 4, 6, ..., 20)? Reply with just the number."
        ),
        expected=["110"],
    ),
    Task(
        id="math_compound",
        category="math",
        prompt=(
            "If you invest $1000 at 5% annual interest compounded annually "
            "for 3 years, what is the final balance? Round to the nearest cent "
            "and reply with a number like 1234.56."
        ),
        expected=["1157.62", "1157.63"],
    ),
    Task(
        id="reasoning_apples",
        category="reasoning",
        prompt=(
            "I start with 3 apples. I give 1 away, buy 5 more, then eat 2. "
            "How many apples do I have? Reply with just the number."
        ),
        expected=["5"],
    ),
    Task(
        id="reasoning_geometric",
        category="reasoning",
        prompt=(
            "What is the next number in the sequence 2, 6, 18, 54, ? "
            "Reply with just the number."
        ),
        expected=["162"],
    ),
    Task(
        id="code_fib",
        category="code",
        prompt=(
            "What is the 10th Fibonacci number? Use the convention "
            "fib(1)=1, fib(2)=1, fib(3)=2, fib(4)=3, ... "
            "Reply with just the number."
        ),
        expected=["55"],
    ),
    Task(
        id="code_vowels",
        category="code",
        prompt=(
            "How many vowels (a, e, i, o, u — case insensitive) are in "
            "the string 'hello world'? Reply with just the number."
        ),
        expected=["3"],
    ),
    Task(
        id="tool_factorial",
        category="tool",
        prompt=(
            "Use the python_exec tool to compute 8! (8 factorial) "
            "and report the result. End your final answer with the number on its own line."
        ),
        expected=["40320"],
        tools=["python_exec"],
        tool_required="python_exec",
    ),
    Task(
        id="tool_pretty",
        category="tool",
        prompt=(
            "Use the pretty_print tool to print the message 'hello' in upper case "
            "with prefix '[bench]'. Then end your final answer with the exact "
            "rendered line on its own line."
        ),
        expected=["[bench] HELLO"],
        tools=["pretty_print"],
        tool_required="pretty_print",
    ),
    Task(
        id="tool_listdir",
        category="tool",
        prompt=(
            "Use the list_dir tool to list the contents of '.' in the fleet "
            "workspace and then say how many entries you saw. End your answer "
            "with the count on its own line."
        ),
        expected=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],  # any digit is fine
        tools=["list_dir"],
        tool_required="list_dir",
    ),
]


# Tasks used only to pre-populate the bank in the memory-augmented run.
# Their solutions reinforce useful habits ("show work", "use python_exec for arithmetic").
WARMUP_TASKS: list[Task] = [
    Task(
        id="warm_math_speed",
        category="math",
        prompt=(
            "A car drives 40 mph for 3 hours. How far? "
            "Reply with just the number."
        ),
        expected=["120"],
    ),
    Task(
        id="warm_reasoning",
        category="reasoning",
        prompt=(
            "Tom has 4 cookies, eats 1, then his sister gives him 3. "
            "How many cookies does Tom have? Reply with just the number."
        ),
        expected=["6"],
    ),
    Task(
        id="warm_tool",
        category="tool",
        prompt=(
            "Use the python_exec tool to compute 7*6 and report the result. "
            "End your answer with the number on its own line."
        ),
        expected=["42"],
        tools=["python_exec"],
        tool_required="python_exec",
    ),
]


# --------------------------------------------------------------------------- verification


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def verify(task: Task, final_text: str, tool_calls_made: list[str]) -> tuple[bool, str]:
    """Return (success, note). Permissive substring + numeric match."""
    if not final_text:
        return False, "empty final text"

    text_lower = final_text.lower().strip()

    if task.tool_required and task.tool_required not in tool_calls_made:
        return False, f"required tool '{task.tool_required}' not called"

    # Numeric tasks: compare last number in the response to expected.
    numeric_exp = [e for e in task.expected if _NUMBER_RE.fullmatch(e.strip())]
    if numeric_exp:
        nums = _NUMBER_RE.findall(final_text)
        if not nums:
            return False, "no number in response"
        last = nums[-1]
        # Allow direct match or float-close match for compound interest.
        try:
            target_vals = [float(e) for e in numeric_exp]
            last_val = float(last)
            for tv in target_vals:
                if abs(tv - last_val) < 0.05:
                    return True, f"numeric match {last}"
            return False, f"got {last}, expected {numeric_exp}"
        except ValueError:
            pass

    # Substring tasks (case insensitive).
    for needle in task.expected:
        if needle.lower() in text_lower:
            return True, f"substring '{needle}'"

    return False, f"none of {task.expected} found"


# --------------------------------------------------------------------------- runners


@dataclass
class RunMetrics:
    task_id: str
    config: str
    success: bool
    note: str
    wall_s: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    tool_calls: list[str]
    final_text: str
    memories_used: int = 0
    memories_distilled: int = 0


def _cost(in_tok: int, out_tok: int) -> float:
    return (in_tok * PRICE_INPUT_PER_M + out_tok * PRICE_OUTPUT_PER_M) / 1_000_000.0


def _collect_tool_calls(state: GraphState) -> list[str]:
    out: list[str] = []
    for m in state.messages:
        for tc in m.tool_calls:
            out.append(tc.name)
    return out


def _final_text(state: GraphState) -> str:
    for m in reversed(state.messages):
        if m.role == "assistant" and m.content:
            return m.content
    return ""


def _count_memory_block_uses(state: GraphState) -> int:
    for m in state.messages:
        if m.role == "system" and m.content and "Relevant past learnings" in m.content:
            return m.content.count("\n   ") // 3 or 0
    return 0


async def run_single(task: Task, llm: FleetLLM, *, bank: ReasoningBank | None) -> RunMetrics:
    agent = Agent(
        llm=llm,
        tools=task.tools,
        max_iters=MAX_ITERS,
        memory_bank=bank,
        memory_k=3 if bank else 0,
    )
    state = GraphState(goal=task.prompt)

    t0 = time.perf_counter()

    # Sum usage across all internal LLM calls by sampling usage after each iter.
    in_tok = 0
    out_tok = 0
    # We'll instrument by wrapping llm.complete to accumulate usage between iters.
    orig_complete = llm.complete
    tracker = {"in": 0, "out": 0}

    async def tracked(messages, tools=None):
        resp = await orig_complete(messages, tools=tools)
        if llm.usage is not None:
            tracker["in"] += int(getattr(llm.usage, "input_tokens", 0) or 0)
            tracker["out"] += int(getattr(llm.usage, "output_tokens", 0) or 0)
        return resp

    llm.complete = tracked  # type: ignore[assignment]
    try:
        final_state = await agent.step(state)
    finally:
        llm.complete = orig_complete  # type: ignore[assignment]

    wall = time.perf_counter() - t0
    in_tok = tracker["in"]
    out_tok = tracker["out"]

    tool_calls = _collect_tool_calls(final_state)
    final = _final_text(final_state)
    ok, note = verify(task, final, tool_calls)
    mem_used = _count_memory_block_uses(final_state) if bank else 0

    return RunMetrics(
        task_id=task.id,
        config="",  # caller fills
        success=ok,
        note=note,
        wall_s=round(wall, 3),
        input_tokens=in_tok,
        output_tokens=out_tok,
        total_tokens=in_tok + out_tok,
        cost_usd=round(_cost(in_tok, out_tok), 6),
        tool_calls=tool_calls,
        final_text=final[:500],
        memories_used=mem_used,
    )


class _RunnableAgent:
    """Wrap an Agent into something matts_run accepts (needs async run(state))."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.memory_bank = agent.memory_bank

    async def run(self, state: GraphState) -> GraphState:
        return await self.agent.step(state)


async def run_matts(
    task: Task,
    llm: FleetLLM,
    *,
    k: int,
    bank: ReasoningBank,
) -> RunMetrics:
    """Run MaTTS k rollouts in parallel and pick the best (judged by verifier)."""
    agent = Agent(
        llm=llm,
        tools=task.tools,
        max_iters=MAX_ITERS,
        memory_bank=bank,
        memory_k=3,
    )
    runnable = _RunnableAgent(agent)

    t0 = time.perf_counter()

    tracker = {"in": 0, "out": 0}
    orig_complete = llm.complete

    async def tracked(messages, tools=None):
        resp = await orig_complete(messages, tools=tools)
        if llm.usage is not None:
            tracker["in"] += int(getattr(llm.usage, "input_tokens", 0) or 0)
            tracker["out"] += int(getattr(llm.usage, "output_tokens", 0) or 0)
        return resp

    llm.complete = tracked  # type: ignore[assignment]
    try:
        states, distilled = await matts_run(runnable, task.prompt, bank=bank, k=k)
    finally:
        llm.complete = orig_complete  # type: ignore[assignment]
    wall = time.perf_counter() - t0

    # Best-of-k: pick the first rollout that verifies; else first one.
    best_state: GraphState = states[0]
    best_ok = False
    best_note = ""
    best_tools: list[str] = []
    best_final = ""
    for st in states:
        tools_made = _collect_tool_calls(st)
        text = _final_text(st)
        ok, note = verify(task, text, tools_made)
        if ok and not best_ok:
            best_state, best_ok, best_note, best_tools, best_final = st, True, note, tools_made, text
            break
        if not best_ok:
            best_state, best_note, best_tools, best_final = st, note, tools_made, text

    in_tok = tracker["in"]
    out_tok = tracker["out"]
    mem_used = _count_memory_block_uses(best_state)

    return RunMetrics(
        task_id=task.id,
        config="",
        success=best_ok,
        note=best_note,
        wall_s=round(wall, 3),
        input_tokens=in_tok,
        output_tokens=out_tok,
        total_tokens=in_tok + out_tok,
        cost_usd=round(_cost(in_tok, out_tok), 6),
        tool_calls=best_tools,
        final_text=best_final[:500],
        memories_used=mem_used,
        memories_distilled=len(distilled),
    )


# --------------------------------------------------------------------------- top level


async def populate_bank(bank: ReasoningBank, llm: FleetLLM) -> int:
    """Run warmup tasks and ingest their trajectories into the bank.

    Returns count of memories added.
    """
    added = 0
    for task in WARMUP_TASKS:
        agent = Agent(
            llm=llm,
            tools=task.tools,
            max_iters=MAX_ITERS,
            memory_bank=None,  # warmup runs don't read memory
            memory_k=0,
        )
        state = GraphState(goal=task.prompt)
        final_state = await agent.step(state)
        # Force success outcome (we just want exemplars in the bank)
        items = await bank.ingest_trajectory(
            list(final_state.messages), task.prompt, outcome="success"
        )
        added += len(items)
        await asyncio.sleep(INTER_CALL_SLEEP_S)
    return added


async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY must be set")

    llm = FleetLLM(BACKEND, MODEL, max_tokens=512, temperature=0.0, api_key=api_key)
    # Use a slightly hotter LLM for MaTTS contrast distill so rollouts diversify.
    matts_llm = FleetLLM(BACKEND, MODEL, max_tokens=512, temperature=0.7, api_key=api_key)

    # Embedder + store shared across configs (reset bank between configs).
    embedder = MiniLMEmbedder()

    all_results: list[RunMetrics] = []

    # --- Config A: baseline k=1, no bank ----------------------------------
    print("\n=== Config A: baseline (k=1, no bank) ===", flush=True)
    for task in TASKS:
        m = await run_single(task, llm, bank=None)
        m.config = "k1_nobank"
        print(
            f"  [{task.id:18s}] ok={m.success} t={m.wall_s:5.2f}s "
            f"in={m.input_tokens} out={m.output_tokens} note={m.note}",
            flush=True,
        )
        all_results.append(m)
        await asyncio.sleep(INTER_CALL_SLEEP_S)
    await asyncio.sleep(INTER_CONFIG_SLEEP_S)

    # --- Config B: k=1 with pre-populated bank ----------------------------
    print("\n=== Config B: k=1 + ReasoningBank ===", flush=True)
    bank_b = ReasoningBank(
        store=InMemoryStore(),
        embedder=embedder,
        induction_llm=llm,
        judge_llm=llm,
        async_writeback=False,
    )
    added_b = await populate_bank(bank_b, llm)
    print(f"  populated bank with {added_b} memories", flush=True)
    for task in TASKS:
        m = await run_single(task, llm, bank=bank_b)
        m.config = "k1_bank"
        print(
            f"  [{task.id:18s}] ok={m.success} t={m.wall_s:5.2f}s "
            f"in={m.input_tokens} out={m.output_tokens} mems={m.memories_used} "
            f"note={m.note}",
            flush=True,
        )
        all_results.append(m)
        await asyncio.sleep(INTER_CALL_SLEEP_S)
    await asyncio.sleep(INTER_CONFIG_SLEEP_S)

    # --- Config C: MaTTS k=2 ---------------------------------------------
    print("\n=== Config C: MaTTS k=2 + bank ===", flush=True)
    bank_c = ReasoningBank(
        store=InMemoryStore(),
        embedder=embedder,
        induction_llm=matts_llm,
        judge_llm=llm,
        async_writeback=False,
    )
    added_c = await populate_bank(bank_c, llm)
    print(f"  populated bank with {added_c} memories", flush=True)
    for task in TASKS:
        m = await run_matts(task, matts_llm, k=2, bank=bank_c)
        m.config = "k2_matts"
        print(
            f"  [{task.id:18s}] ok={m.success} t={m.wall_s:5.2f}s "
            f"in={m.input_tokens} out={m.output_tokens} distilled={m.memories_distilled} "
            f"note={m.note}",
            flush=True,
        )
        all_results.append(m)
        await asyncio.sleep(INTER_CALL_SLEEP_S * 2)  # MaTTS uses 2-4x tokens
    await asyncio.sleep(INTER_CONFIG_SLEEP_S)

    # --- Config D: MaTTS k=4 ---------------------------------------------
    print("\n=== Config D: MaTTS k=4 + bank ===", flush=True)
    bank_d = ReasoningBank(
        store=InMemoryStore(),
        embedder=embedder,
        induction_llm=matts_llm,
        judge_llm=llm,
        async_writeback=False,
    )
    added_d = await populate_bank(bank_d, llm)
    print(f"  populated bank with {added_d} memories", flush=True)
    for task in TASKS:
        m = await run_matts(task, matts_llm, k=4, bank=bank_d)
        m.config = "k4_matts"
        print(
            f"  [{task.id:18s}] ok={m.success} t={m.wall_s:5.2f}s "
            f"in={m.input_tokens} out={m.output_tokens} distilled={m.memories_distilled} "
            f"note={m.note}",
            flush=True,
        )
        all_results.append(m)
        await asyncio.sleep(INTER_CALL_SLEEP_S * 3)
    await asyncio.sleep(INTER_CONFIG_SLEEP_S)

    # --- write artifacts --------------------------------------------------
    raw_path = RESULTS_DIR / "results.json"
    with raw_path.open("w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nWrote raw results → {raw_path}", flush=True)

    # aggregate summary
    summary: dict[str, dict[str, Any]] = {}
    for cfg in ("k1_nobank", "k1_bank", "k2_matts", "k4_matts"):
        rows = [r for r in all_results if r.config == cfg]
        if not rows:
            continue
        n = len(rows)
        succ = sum(1 for r in rows if r.success)
        in_t = sum(r.input_tokens for r in rows)
        out_t = sum(r.output_tokens for r in rows)
        cost = sum(r.cost_usd for r in rows)
        wall = sum(r.wall_s for r in rows)
        tool_attempts = sum(1 for r in rows if r.tool_calls)
        # by category
        by_cat: dict[str, dict[str, int]] = {}
        for r in rows:
            cat = next((t.category for t in TASKS if t.id == r.task_id), "?")
            d = by_cat.setdefault(cat, {"n": 0, "succ": 0})
            d["n"] += 1
            d["succ"] += 1 if r.success else 0
        summary[cfg] = {
            "n_tasks": n,
            "accuracy": round(succ / n, 3),
            "success_count": succ,
            "input_tokens": in_t,
            "output_tokens": out_t,
            "total_tokens": in_t + out_t,
            "cost_usd": round(cost, 5),
            "wall_s_total": round(wall, 2),
            "wall_s_avg": round(wall / n, 3),
            "tool_call_runs": tool_attempts,
            "by_category": {
                cat: {**d, "accuracy": round(d["succ"] / d["n"], 3)}
                for cat, d in by_cat.items()
            },
        }

    sum_path = RESULTS_DIR / "summary.json"
    with sum_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary → {sum_path}", flush=True)

    print("\n=== SUMMARY ===")
    for cfg, s in summary.items():
        print(
            f"  {cfg}: acc={s['accuracy']:.2f} ({s['success_count']}/{s['n_tasks']}) "
            f"in={s['input_tokens']} out={s['output_tokens']} "
            f"${s['cost_usd']:.4f} {s['wall_s_total']:.1f}s"
        )


if __name__ == "__main__":
    asyncio.run(main())
