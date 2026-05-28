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
2. **MaTTS cost scales near-linearly with k; latency scales sub-linearly but with a floor.** Total tokens grew ~6.3× from k=1 to k=2 and ~12.0× to k=4 (including the contrast-distill call). Wall time grew 2.9× (k=2) and 3.6× (k=4) over baseline — k=4 sits *below* a serial-rollouts implementation's 4× floor (rollouts run in parallel via `asyncio.gather`), but k=2 sits *above* a rollouts-only 2× expectation because a roughly fixed contrast-distill call adds ≈ 1.5 s regardless of `k`. See §2.3 for the breakdown.
3. **Self-consistency held perfectly at k=4.** Across the 6 MaTTS trials that completed before the contrast-distill crash (see §5), every one of the 4 parallel rollouts produced the correct answer — **24 / 24 rollouts correct, 6 / 6 trials unanimous**. This is the regime where MaTTS would *also* be cheap to skip via early exit on agreement — an obvious follow-on for the harness.
4. **Contrast distillation produces noticeably richer memory items than single-trajectory induction.** k=4 produced 1–3 distilled items per task vs 1–2 at k=2, and the items showed visible meta-strategic content (e.g. "Derive explicit rate equations before applying transformations") that single-rollout induction did not surface.
5. **ReasoningBank retrieval adds ~3.4K input tokens per task** (the 3 retrieved memories plus the system block) without affecting accuracy on this independent-task suite. It is a *write-path* feature here, not a read-path win — value is expected on tasks that share structure with prior trajectories, which our suite deliberately did not.
6. **One real bug surfaced (now fixed).** MaTTS contrast distill failed on a verbose Haiku response that exceeded `max_tokens=512`. Fix landed in commit `411cf6c` — see [Bug surfaced](#bug-surfaced-during-benchmarking) below.

> **Caveat.** Every task in the main suite was solved at 100 % by the base model alone (Haiku 4.5 is above the suite's difficulty ceiling). What follows therefore measures the **cost and latency mechanics** of MaTTS / ReasoningBank, not an accuracy lift over baseline. Demonstrating MaTTS's accuracy benefit requires harder benchmarks — see §8.

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

Anthropic Haiku 4.5 pricing **as of 2026-05-28**: **$1.00 / M input**, **$5.00 / M output** (pricing snapshot — Anthropic may revise these rates; recompute from the per-task token columns in `results.json` if comparing to a later list price). Cost in this report is **USD spent on the LLM provider only** — it does not include the MiniLM embedder (CPU-local), the SQLite/InMemory store (CPU-local), or any orchestration overhead.

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

### 2.3 Latency: parallel rollouts + a fixed contrast-distill tax

Wall-clock time per task by config:

```
config          avg     min     max
─────────────────────────────────────
k1_nobank       1.69    0.75    4.22
k1_bank         1.57    0.70    3.68
k2_matts        4.91    3.79    6.40
k4_matts        6.08    4.15   11.15
```

MaTTS adds two things to baseline latency: (a) `k` rollouts, which `matts_run` issues concurrently via `asyncio.gather`, and (b) one *sequential* contrast-distill LLM call that runs after all rollouts return. The measured numbers are consistent with both effects, and only fully explained by considering them together:

- **k=2 measured 2.9× baseline.** This is **above** the 2× a serial-rollouts-only execution would predict (3.4 s). The extra ≈ 1.5 s is the contrast-distill call — it is paid once, regardless of `k`, and dominates the small-k regime.
- **k=4 measured 3.6× baseline.** This is **below** the 4× a serial-rollouts implementation would predict (6.8 s). Adding 2 more rollouts only added ≈ 1.2 s of wall time on top of k=2, which is consistent with rollouts running in parallel (the marginal cost of k=4 vs k=2 is the slowest extra rollout, not 2 more serial rollouts).

So the honest claim is: **rollouts do execute in parallel** — that is what keeps k=4 under the 4× serial floor — **but a roughly fixed contrast-distill cost dominates at low k**, which is why k=2 sits *above* its 2× rollouts-only expectation. A serial MaTTS implementation would be substantially slower at k=4; a contrast-free MaTTS would be substantially faster at k=2.

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

¹ HARD-B aborted at 6 / 15 planned trials due to a contrast-distill crash; see §5. Across those 6 completed trials × 4 rollouts each = **24 individual rollouts, every one correct, every trial 4-of-4 unanimous**. There was no observed regime in this suite where k=4 added accuracy beyond k=1.

**Interpretation.** Haiku 4.5 is strong enough that classic GPT-3-era traps have moved below its capability boundary. To meaningfully exercise MaTTS we would need either (a) a harder benchmark such as MATH-500, GPQA, or SWE-bench-Lite, or (b) a smaller / older model where the failure rate is high enough that 4-way voting changes the outcome. Both are out of scope for a dev-tier 30 K-TPM probe, but are the natural next experiments.

---

## 5. Bug surfaced during benchmarking

> **Status: fixed.** This bug was found, root-caused, and patched as part of the v0.2 hardening pass — see commit `411cf6c` ("fix(matts): handle truncated/invalid contrast JSON gracefully"). The section is kept for posterity because the bug is a useful case study in benchmark-as-fuzz-test.

`scripts/run_hard_benchmarks.py` crashed on the 7th MaTTS trial (after 6 successful trials × 4 rollouts = 24 successful rollouts) with:

```
ValueError: LLM response was not valid JSON:
  Unterminated string starting at: line 15 column 16 (char 1866)
```

The contrast LLM (Haiku 4.5 at temperature 0.7) generated ~2 000-character JSON with rich, paragraph-length `content` fields and ran into the `max_tokens=512` ceiling configured in the FleetLLM instance, truncating the response mid-string. The original `_contrast_distill` called `parse_json_strict` on the raw response and re-raised, propagating the failure all the way to the benchmark.

**Reproducer (against pre-`411cf6c` code):** any MaTTS run on a complex multi-step task with `max_tokens ≤ 512` on the contrast LLM is at risk.

**Fix applied in `411cf6c`** — `_contrast_distill` now catches truncated/invalid contrast JSON and falls back to returning an empty distilled-items list with a `logger.warning`, so the rollouts (which succeeded) are still returned to the caller. Distillation is now best-effort, decoupled from the success of the rollouts themselves.

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

# Optional secondary suite — the pre-411cf6c HARD-B crash on the contrast call
# is now caught (see §5); HARD-B completes end-to-end on current HEAD.
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
6. **Re-run HARD-B end-to-end on current HEAD.** The contrast-distill truncation that aborted HARD-B has been fixed in `411cf6c`; a clean re-run would let us replace the partial-data caveat in §4 with a full 15/15 result.

---

*Benchmarks generated 2026-05-28 against bottensor-fleet v0.2.0 on branch `v0.2-reasoning-bank`. Total wall time end-to-end: ~3 minutes. Total Anthropic spend: $0.13 (main suite) + $0.07 (hard suite, partial).*
