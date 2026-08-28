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
