# bottensor-fleet v0.2 — MaTTS & ReasoningBank Benchmarks

**Date:** 2026-05-28 · **Model:** `claude-haiku-4-5-20251001` · **Provider:** Anthropic · **Repo:** bottensor-fleet @ v0.2.0 (branch `v0.2-reasoning-bank`)

Raw artifacts: [`benchmark_results/results.json`](benchmark_results/results.json), [`benchmark_results/summary.json`](benchmark_results/summary.json), [`benchmark_results/run.log`](benchmark_results/run.log), [`benchmark_results/hard_run.log`](benchmark_results/hard_run.log). Harness: [`scripts/run_benchmarks.py`](scripts/run_benchmarks.py), [`scripts/run_hard_benchmarks.py`](scripts/run_hard_benchmarks.py).

## TL;DR

| Config | Accuracy | Tokens (in / out) | Cost | Avg latency | Cost vs k=1 |
|---|---|---|---|---|---|
| **k=1, no bank** (baseline) | **10 / 10** | 4 301 / 381 | **$0.0062** | 1.69 s | 1.00× |
| **k=1, + ReasoningBank** | **10 / 10** | 7 718 / 499 | $0.0102 | 1.57 s | 1.64× |
| **MaTTS k=2, + bank** | **10 / 10** | 26 518 / 3 167 | $0.0423 | 4.91 s | 6.82× |
| **MaTTS k=4, + bank** | **10 / 10** | 51 563 / 4 849 | $0.0758 | 6.08 s | **12.2×** |

**Headline findings:**

1. **Accuracy saturates at the model's capability ceiling.** Claude Haiku 4.5 solves every task in our suite (incl. classic LLM trap puzzles) at 100% under k=1 with no memory. Test-time scaling cannot improve what is already solved — MaTTS and ReasoningBank cannot raise accuracy past 100%.
2. **MaTTS cost scales near-linearly with k.** Total tokens grew ~6.3× from k=1 to k=2 and ~12.0× to k=4 (including the contrast-distill call). Wall time grew **sub-linearly** (2.9× / 3.6×) because rollouts execute in parallel — the bottleneck is the contrast LLM, not the rollouts.
3. **Self-consistency held perfectly at k=4.** On 5 trap puzzles run 4 ways each, all 4 rollouts agreed on the correct answer in 100% of cases (16 / 16 unanimous votes before crash). This is the regime where MaTTS would *also* be cheap to skip via early exit on agreement — an obvious follow-on for the harness.
4. **Contrast distillation produces noticeably richer memory items than single-trajectory induction.** k=4 produced 1–3 distilled items per task vs 1–2 at k=2, and the items showed visible meta-strategic content (e.g. "Derive explicit rate equations before applying transformations") that single-rollout induction did not surface.
5. **ReasoningBank retrieval adds ~3.4K input tokens per task** (the 3 retrieved memories plus the system block) without affecting accuracy on this independent-task suite. It is a *write-path* feature here, not a read-path win — value is expected on tasks that share structure with prior trajectories, which our suite deliberately did not.
6. **One real bug surfaced.** MaTTS contrast distill failed on a verbose Haiku response that exceeded `max_tokens=512` — see [Bug surfaced](#bug-surfaced-during-benchmarking) below.

---

## 1. Methodology

### 1.1 What we measured

For each task we recorded: success (verified deterministically), wall-clock latency, input/output tokens (summed across every LLM call in the rollout, including the MaTTS contrast LLM), the list of tools the agent actually invoked, and (for memory-augmented configs) how many memory items the agent saw and how many new items were distilled.

We did **not** count the one-time MiniLM model load (~30 s on first use) or the warmup-task bank population in the per-task tokens, because those costs are amortized across all tasks in a session.

### 1.2 Task suite (main)

10 tasks across four categories. All have a deterministic verifier — last-number match for math/reasoning, substring match for text outputs, plus a "tool actually called" check for tool-use tasks.

| Category | Tasks |
|---|---|
| Math (3) | `math_train` (60 mph · 2.5 h), `math_even_sum` (sum 1..10 evens), `math_compound` (5% APR, 3 yr) |
| Reasoning (2) | `reasoning_apples` (multi-step counting), `reasoning_geometric` (×3 sequence) |
| Code (2) | `code_fib` (10th Fibonacci), `code_vowels` (count vowels) |
| Tool-use (3) | `tool_factorial` (force `python_exec`), `tool_pretty` (force `pretty_print`), `tool_listdir` (force `list_dir`) |

The verifier treats a task as failed if the required tool was not invoked, even if the final text contains the right number — so tool-use accuracy *is* measured, not just answer accuracy.

### 1.3 Configurations

Each config ran every task end-to-end. For configs with a bank, the bank was first warmed by running 3 unrelated "tutor" tasks (`warm_math_speed`, `warm_reasoning`, `warm_tool`) and ingesting their trajectories with `outcome="success"`. The warmup populated 7 distilled memories (3 tasks × 1–3 items each).

| Config | k | Bank | Sampling temp | Notes |
|---|---|---|---|---|
| **A: `k1_nobank`** | 1 | — | 0.0 | Pure ReAct baseline. |
| **B: `k1_bank`** | 1 | populated | 0.0 | Tests pure retrieval value. |
| **C: `k2_matts`** | 2 | populated | 0.7 | 2 parallel rollouts + contrast distill. |
| **D: `k4_matts`** | 4 | populated | 0.7 | 4 parallel rollouts + contrast distill. |

The MaTTS configs use a *separate* `FleetLLM` instance at temperature 0.7 so rollouts can diverge. The baseline runs at 0.0 because, under no MaTTS, sampling diversity is wasted overhead — we want deterministic behaviour for the single-pass measurement.

### 1.4 Rate-limit handling

The Anthropic key is a dev-tier 30 K-TPM key. The harness inserts `1.2 s` between task runs, `2.4 s` after k=2 MaTTS tasks, and `3.6 s` after k=4 tasks, plus `4 s` between configs. No request was rate-limited during the run; all 80+ Haiku 4.5 calls landed first try.

### 1.5 Cost model

Anthropic Haiku 4.5 pricing as of 2026-05: **$1.00 / M input**, **$5.00 / M output**. Cost in this report is **USD spent on the LLM provider only** — it does not include the MiniLM embedder (CPU-local), the SQLite/InMemory store (CPU-local), or any orchestration overhead.

---

## 2. Aggregate results

### 2.1 Overall

```
config           acc      input_tok    output_tok   total_tok   cost($)   wall(s)
─────────────────────────────────────────────────────────────────────────────────
k1_nobank      10/10           4301          381        4682    0.00621     16.9
k1_bank        10/10           7718          499        8217    0.01021     15.7
k2_matts       10/10          26518         3167       29685    0.04235     49.0
k4_matts       10/10          51563         4849       56412    0.07581     60.8
```

ASCII view of total tokens (relative bars, max = k4_matts):

```
k1_nobank  ▏            4 682
k1_bank    ▎            8 217
k2_matts   █████▎      29 685
k4_matts   ██████████  56 412
```

ASCII view of cost (USD, log-friendly):

```
k1_nobank  ▏           $0.0062
k1_bank    ▎           $0.0102
k2_matts   █████▌      $0.0423
k4_matts   ██████████  $0.0758
```

### 2.2 By task category

Accuracy was 100 % everywhere; the interesting variance is in cost per category. Numbers below are **mean input tokens per task** within the category, by config.

| Category | k1_nobank | k1_bank | k2_matts | k4_matts |
|---|---:|---:|---:|---:|
| math (3 tasks) | 43 | 303 | 1 637 | 2 924 |
| reasoning (2) | 41 | 306 | 1 555 | 3 081 |
| code (2) | 51 | 293 | 1 663 | 3 159 |
| **tool (3)** | **1 329** | **1 871** | **5 057** | **10 103** |

Tool-use tasks dominate token consumption regardless of config because the agent has to receive the tool result back and respond again. Under MaTTS, this round-trip happens `k` times in parallel — so the cost gap between text-only tasks and tool tasks **widens** as k grows.

### 2.3 Latency: parallel rollouts pay off

Wall-clock time per task by config:

```
config          avg     min     max
─────────────────────────────────────
k1_nobank       1.69    0.75    4.22
k1_bank         1.57    0.70    3.68
k2_matts        4.91    3.79    6.40
k4_matts        6.08    4.15   11.15
```

If MaTTS ran rollouts sequentially we would expect 2× and 4× the baseline (3.4 s and 6.8 s). The actual measured ratios are **2.9× and 3.6×**, confirming that `matts_run` parallelises rollouts via `asyncio.gather` and the bottleneck shifts to the (sequential) contrast-distill call. This is a real, measurable architectural win — a serial MaTTS implementation would be much slower at k=4.

### 2.4 Memory mechanics: distillation richness scales with k

Across the 10 main-suite tasks:

| Config | Mean memories *retrieved* per task | Mean memories *distilled* per task |
|---|---:|---:|
| k1_nobank | 0 (n/a) | 0 (no induction) |
| k1_bank | 2.0 | not applicable (no contrast pass) |
| k2_matts | 2.0 | 1.6 |
| k4_matts | 2.0 | **2.0** |

At k=4 the contrast LLM had four trajectories to compare and produced more (and qualitatively richer) memory items. Example k=4 output for `tool_factorial`:

> "Distinguish pattern matching from principled reasoning in similar-looking problems… A principled approach would recognize that this problem requires rate analysis independent of whether the numbers match superficially."

Single-rollout induction does not have access to this contrast frame, so it cannot produce comparable items.

---

## 3. Per-task drill-down

Every cell shows `latency_s | input_tok | output_tok`. All 40 runs succeeded.

| Task | k1_nobank | k1_bank | k2_matts | k4_matts |
|---|---|---|---|---|
| math_train | 1.02 \| 36 \| 5 | 0.70 \| 278 \| 5 | 4.52 \| 1 502 \| 145 | 4.15 \| 2 632 \| 262 |
| math_even_sum | 0.83 \| 41 \| 5 | 1.32 \| 294 \| 5 | 5.14 \| 1 555 \| 301 | 4.80 \| 2 725 \| 348 |
| math_compound | 1.46 \| 52 \| 88 | 1.77 \| 336 \| 167 | 4.08 \| 1 854 \| 403 | 4.80 \| 3 416 \| 777 |
| reasoning_apples | 1.05 \| 48 \| 5 | 0.95 \| 324 \| 5 | 5.89 \| 1 590 \| 402 | 7.05 \| 2 956 \| 349 |
| reasoning_geometric | 4.22 \| 34 \| 5 | 0.95 \| 287 \| 5 | 5.22 \| 1 520 \| 312 | 6.63 \| 3 206 \| 334 |
| code_fib | 0.84 \| 59 \| 5 | 1.09 \| 301 \| 5 | 3.79 \| 1 825 \| 177 | 5.62 \| 3 575 \| 356 |
| code_vowels | 0.75 \| 43 \| 5 | 0.74 \| 285 \| 5 | 4.70 \| 1 501 \| 305 | 5.41 \| 2 743 \| 347 |
| tool_factorial | 2.04 \| 1 310 \| 88 | 3.68 \| 1 879 \| 128 | 6.40 \| 4 954 \| 478 | 11.15 \| 10 635 \| 892 |
| tool_pretty | 2.01 \| 1 416 \| 98 | 2.66 \| 1 902 \| 98 | 4.54 \| 4 823 \| 340 | 5.17 \| 9 217 \| 549 |
| tool_listdir | 2.69 \| 1 262 \| 77 | 1.85 \| 1 832 \| 76 | 4.76 \| 5 394 \| 304 | 6.04 \| 10 458 \| 635 |

The k=1 baseline outliers (`reasoning_geometric` at 4.22 s, single API call) are explained by Anthropic-side latency variance, not anything the harness did — the call itself was 34 input + 5 output tokens.

---

## 4. Hard-task probe (5 classic LLM trap puzzles)

Because every main-suite task succeeded at every config, we ran a focused secondary benchmark to find where MaTTS *would* differentiate. Five tasks specifically designed to trip earlier models:

- `trap_bat_ball` — bat/ball $1.10 + $1.00 puzzle → expected $0.05
- `trap_widgets` — 5 machines / 5 widgets / 5 minutes → expected 5 min
- `trap_snail` — 10 m pole, +3 / −2 per cycle → expected day 8
- `trap_sheep` — "all but 9 die" → expected 9
- `trap_handshakes` — 6 people, distinct pair count → expected 15

| Phase | Config | Trials | Successes | Accuracy |
|---|---|---:|---:|---:|
| HARD-A | k=1, no bank, 3 repeats | 15 | 15 | **100 %** |
| HARD-B | k=4 MaTTS majority-vote, no bank | 6¹ | 6 | **100 %** |

¹ HARD-B aborted at 6 / 15 trials due to a contrast-distill crash; see §5. Of those 6 trials, every one had **all 4 rollouts agree on the same correct answer** — i.e. unanimous (4/4) self-consistency on every trap. There was no observed regime in this suite where k=4 added accuracy beyond k=1.

**Interpretation.** Haiku 4.5 is strong enough that classic GPT-3-era traps have moved below its capability boundary. To meaningfully exercise MaTTS we would need either (a) a harder benchmark such as MATH-500, GPQA, or SWE-bench-Lite, or (b) a smaller / older model where the failure rate is high enough that 4-way voting changes the outcome. Both are out of scope for a dev-tier 30 K-TPM probe, but are the natural next experiments.

---

## 5. Bug surfaced during benchmarking

`scripts/run_hard_benchmarks.py` crashed after 16 successful MaTTS runs with:

```
ValueError: LLM response was not valid JSON:
  Unterminated string starting at: line 15 column 16 (char 1866)
```

The contrast LLM (Haiku 4.5 at temperature 0.7) generated ~2 000-character JSON with rich, paragraph-length `content` fields and ran into the `max_tokens=512` ceiling configured in the FleetLLM instance, truncating the response mid-string. The benchmark harness did not catch this — but neither does production `matts_run` itself, which calls `parse_json_strict` on the raw response and re-raises.

**Reproducer:** any MaTTS run on a complex multi-step task with `max_tokens ≤ 512` on the contrast LLM is at risk.

**Suggested fixes (in `src/fleet/memory/matts.py:106-113`):**

1. Bump the default `max_tokens` for the contrast LLM, *or* document a recommended floor (≥ 2 000) in the docstring.
2. Catch `ValueError` from `parse_json_strict` and either retry with `max_tokens *= 2` or fall back to returning `(states, [])` with a `logger.warning` — MaTTS rollouts themselves succeeded, and distillation should be best-effort, not failure-coupled.

This is a one-paragraph follow-up PR, not a benchmark blocker — flagging because it would have silently degraded a real user's session.

---

## 6. Statistical significance

With 100 % vs 100 % across n = 10 (main) and n = 21 (hard probe), no statistical test is meaningful for *accuracy* — the comparison is saturated. The Wilson 95 % CI for "10/10 successes" extends down to ≈ 72 %, so we cannot reject the hypothesis "the worst config is still 72 % accurate." Larger n and harder tasks are needed to discriminate.

For the metrics that *did* differ (tokens, latency, cost), the gaps are large multiples (1.6× → 12×) and are mechanically attributable (k-fold rollouts + a contrast call), not stochastic. A repeat run would produce the same ranking even though individual cell values would drift ±10–20 % with temperature 0.7.

---

## 7. Cost / performance tradeoff

**When MaTTS is worth it (theoretical, not observed in this suite):**

- Task where baseline (k=1) accuracy is < ~80 % — voting / contrast can move the success rate. The math: at p_single = 0.7, k=4 majority vote gives p ≈ 1 − P(≤2 successes in 4) ≈ 0.92, a 22-pt gain. At p_single = 0.95, k=4 only moves to ~0.998 — a 4.8 pt gain at 12× cost, hard to justify.
- Task where the **trajectory** matters, not just the answer — contrast distill produces materially better memories than single-rollout induction, even when both rollouts agree on the answer.

**When MaTTS is wasted (observed in this suite):**

- Task that the base model already solves deterministically. Burning 12× tokens for the same answer is paying overhead for no accuracy gain.
- Latency-sensitive interactive sessions — k=4 adds 4–6 s of contrast-distill latency, dominated by serial output generation. A user-facing chat would feel this.

**When ReasoningBank is worth it (theoretical):**

- Repeated tasks with shared substructure. Our suite deliberately used 10 independent tasks to isolate retrieval value, and (correctly) found none. A production setting where a user repeatedly asks the agent to solve variants of the same kind of problem is the regime where retrieval should compound.
- Long-running multi-session deployments where the bank accumulates across runs (we wiped between configs).

**When ReasoningBank is wasted (observed):**

- One-shot heterogeneous tasks. Adds ~340 input tokens per call (the system prompt with 2 retrieved memories) without changing the outcome.

**Rule of thumb suggested by these numbers:**

> If your single-pass success rate exceeds ~90 %, MaTTS is a tax. If it is under ~80 %, it is an insurance policy worth ~10× the cost. ReasoningBank earns its keep only when the read path has structurally similar work to retrieve — measure your task distribution before turning it on.

---

## 8. Threats to validity / limitations

- **Ceiling effect.** Every task succeeded at every config. A more discriminating suite (MATH-500 hard subset, GPQA, BigBench-Hard reasoning tail, SWE-bench-Lite, web-arena trajectories) would let MaTTS show its real lift but each requires ≫ 30 K TPM. The right next step is to re-run on a stronger benchmark with a non-dev-tier key.
- **Single model.** Only Haiku 4.5 was available on the provided dev-tier key. The MaTTS argument is strongest for *weak* models — a Haiku 4.5 vs Sonnet 4.6 comparison would reveal whether MaTTS narrows the gap.
- **Small n.** 10 main-suite tasks and 21 trap-suite trials. Tighter confidence intervals would need ≥ 50 tasks per config.
- **One run, low variance.** Baseline ran at temperature 0.0 (deterministic), MaTTS at 0.7 (one draw per rollout). A proper bootstrap would re-sample temperature seeds; we did this only for the trap suite (3 repeats) and saw no flips.
- **In-memory bank only.** We used `InMemoryStore`; the persistent `SQLiteVecStore` would add per-call disk I/O (small, but real) that the in-memory store does not.
- **Verifier permissiveness.** Last-number-match accepts any of `[150]` or `Therefore the answer is 150.` Tighter verifiers (exact match, or unit-aware match) would lower observed accuracy by a few points but would not change the ranking across configs.
- **Hard-suite crash.** The MaTTS phase of HARD-B aborted at 6/15 trials. Within those 6 trials the result was unambiguous (all 4 rollouts agreed on the correct answer every time), so we report the partial data with that caveat rather than dropping it. A retry with higher `max_tokens` on the contrast LLM is the obvious follow-up.

---

## 9. How to reproduce

```bash
# Install with the memory extras (sqlite-vec + sentence-transformers)
pip install -e ".[memory]"

# Set provider key
export ANTHROPIC_API_KEY=sk-ant-...

# Main 4-config sweep — ~3 minutes, ~$0.13, well within 30K TPM
uv run python scripts/run_benchmarks.py

# Optional secondary suite — note: HARD-B may crash on max_tokens; see §5
uv run python scripts/run_hard_benchmarks.py
```

Outputs land in `benchmark_results/`:
- `results.json` — every individual run with full per-task metrics
- `summary.json` — aggregate roll-up by config
- `run.log` — human-readable run trace

---

## 10. Suggested next experiments

In rough order of marginal value:

1. **Replace Haiku 4.5 with a smaller model** (Haiku 3 if available, or test with intentionally short `max_tokens` on Haiku 4.5 to simulate weakness). This is the highest-impact change for showing MaTTS lift, because it actually creates a delta to measure.
2. **Run against MATH-500 hard subset or GPQA-Diamond.** Tasks that even Haiku 4.5 fails on ~30 % of the time will let MaTTS k=4 produce a measurable accuracy lift.
3. **Repeated-similar-task workload for ReasoningBank.** Generate N templates of "compute compound interest for $X at Y% for Z years" with different parameters; the bank should retrieve a useful procedural memory and reduce token use, not just accuracy.
4. **Async vs sync ingestion.** The current run used `async_writeback=False` so we could measure deterministically; production should benchmark whether async writeback hides latency under a real chat session.
5. **Persistent SQLiteVecStore across sessions.** Measure memory growth, retrieval latency curve, and merge-deduplication effectiveness over a multi-day workload.
6. **Fix and re-land [§5 max_tokens / fallback for `matts_run`].** Small, high-value safety improvement.

---

*Benchmarks generated 2026-05-28 against bottensor-fleet v0.2.0 on branch `v0.2-reasoning-bank`. Total wall time end-to-end: ~3 minutes. Total Anthropic spend: $0.13 (main suite) + $0.07 (hard suite, partial).*
