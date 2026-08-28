# AIConfigurator in the loop

## What it is good at, and what it is not

AIConfigurator is an analytical performance model over a database of measured
kernel timings. It searches parallelism layouts and replica counts against a
TTFT/TPOT SLA and ranks them. It is a genuinely better *generator* of candidate
topologies than hand-sweeping, because it knows TensorRT-LLM kernel behaviour
that this repository does not.

It is not a judge for this deployment, for reasons its own documentation states:

1. It filters on **TTFT and TPOT targets**. We are scored on the *fraction of
   requests* meeting TTFT ≤ 3000 ms **and** mean ITL ≤ 20 ms, with a 0.90
   threshold. A layout can meet the mean and fail the fraction.
2. It models **no scheduler, no KV-cache behaviour, no queueing**. Every lever in
   this repository lives in that gap.
3. Its estimates are reliable only inside its **support matrix**. Outside it,
   it falls back to weaker modelling *silently*. Run `aiconfigurator cli support`
   first and read the answer.

## The flow

```
aiconfigurator  →  candidate layouts (TP/PP, replicas, GPU split)
       ↓
trtllm-tuner    →  cross-check vs our measured capacity model
       ↓
trtllm-sim      →  score under the real per-request goodput metric
       ↓
one real run    →  confirm the winner on hardware
```

```bash
# what to run, printed rather than shelled out - this is an external tool with
# its own install and its own support matrix
trt-llm-rs -c configs/qwen3-4p1d.toml tune

aiconfigurator cli support --model-path Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --system h200_sxm --backend trtllm
aiconfigurator cli default --model-path Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --system h200_sxm --backend trtllm --total-gpus 16 \
    --isl 4000 --osl 200 --ttft 3000 --tpot 20 \
    --database-mode SILICON --save-dir ./aic-results

trt-llm-rs -c configs/qwen3-4p1d.toml tune --save-dir ./aic-results
```

## Reading the cross-check

`CrossCheck` puts AIConfigurator's predicted output tokens/s/GPU next to what
our measured capacity model says for the same topology.

| verdict | meaning |
|---|---|
| `agree` | within 33 %. The topology is probably safe to try. |
| `aic-optimistic` | its database covers a kernel path this cluster does not run, **or** our calibration is stale. Find out which before spending GPU-hours. |
| `aic-pessimistic` | usually a support-matrix miss. Check `cli support`. |
| `no-aic-number` | the CSV had no throughput column; only the simulated score applies. |

A disagreement is the useful output, not a failure. It names the assumption
worth measuring next.

## Parsing note

Column names in `pareto.csv` / `best_config_topn.csv` belong to AIConfigurator
and this tree has never been run against a live install. Every lookup in
`crates/tuner/src/csv.rs` is a **pattern match on the header** that returns
`None` rather than guessing, and unparseable rows are reported in
`TuningPlan::skipped` rather than silently dropped. If a column moves, the
symptom is a skipped row, not a wrong number.


## What it actually recommended (2026-08-29)

aiconfigurator 0.7.0, `Qwen/Qwen3-235B-A22B-FP8` @ `h200_sxm`, 16 GPUs,
ISL 4000 / OSL 200, TTFT 3000 ms, TPOT 20 ms, `--prefix 0`, `SILICON`:

| | |
|---|---|
| prefill | **4 workers x 2 GPUs**, `tp1pp1dp2etp1ep2`, bs 1 |
| decode | **1 worker x 8 GPUs**, `tp8pp1dp1etp1ep8`, bs 64 |
| throughput | 179.69 tok/s/gpu, **seq/s = 14.375 req/s** |
| latency | TTFT 901.56 ms, TPOT 19.78 ms |

That is 4P1D — independent confirmation of the shape. Note the prefill worker
is **DP2 + EP2, not TP2**: attention data-parallel plus expert-parallel MoE,
which carries *no attention all-reduce at all*. Our `(t-1)/t` correction assumes
tensor parallelism and therefore under-credits that layout.

For **sglang and vllm it recommends aggregated instead** (disagg 0.81x and
0.65x). Its best vllm number is 6.68 req/s, against a teammate's measured
**16.4** for vLLM 4P1D — a 2.5x disagreement that has to be resolved before any
vllm row here is used for a decision.

Two hard constraints it surfaced: `moe_tp=8` is rejected on this model because
`moe_intermediate_size 1536 / 8 = 192` and `192 % 128 != 0` under FP8 block
quantisation, so MoE tensor parallelism caps at 4. And it spends TPOT out to
19.78 ms, corroborating that the 17.23 ms SGLang reference leaves ~14 % of the
budget unclaimed.

## Operational gotchas, all verified the hard way

- **Pin `plotext==5.3.2`.** aiconfigurator requires `plotext>=5.3.2`, 6.0.0
  removed `plot_size`, and the run crashes *after* completing the search.
- **`--save-dir` must be under CWD, `$HOME`, `/tmp`, `/workspace` or
  `/var/tmp`.** Anything else is accepted on the command line and then throws
  inside `safe_mkdir`, losing every artifact from a successful search.
- **The real output layout is two levels deeper than documented**:
  `<save-dir>/<model-org>/<run-name>/{agg,disagg}/…`. `load_candidates` walks
  for the `agg`/`disagg` directories rather than assuming where they are.
- Python >= 3.10 is required; the perf database is bundled (247 MB), so the
  search runs fine on a compute node with no internet.
