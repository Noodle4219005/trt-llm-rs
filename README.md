# trt-llm-rs

A Rust control plane for disaggregated LLM serving: it takes over the
*policy* in NVIDIA Dynamo and TensorRT-LLM — which requests are admitted and
when — and leaves their mechanism where it is.

Scope narrowed deliberately and it is worth stating up front, because an
earlier version of this file claimed the project rewrites "everything that
decides which tokens run and when". It does not. ADR 0034: the Rust scheduler
is substituted for TensorRT-LLM's capacity scheduler through the duck-typed
`RequestScheduler` seam, and the ~298-line overlap executor loop is left alone
because that loop is worth +67% ITL (job 312007). ADR 0035 put the size of the
finished adapter at four items, not 2,837 new lines.

Target workload: Qwen3-235B-A22B-Instruct FP8, ISL 4000 / OSL 200, 16×H200,
closed loop.

The bar to beat is **vLLM 4P1D, goodput 16.4** — a teammate's run. No artifact
for it has reached this tree, so it is used here as an *external reference
point* and nothing else: no calibration constant in this repository is
back-derived from it. The constants that are fitted come from runs whose raw
artifacts exist, and each one names its run. Notably 16.4 is **above** what the
calibrated model predicts for a 16-GPU 8P/8D split (≈14.3), so the vLLM decode
side is holding either more concurrency or a lower ITL than the SGLang run this
model was fitted to. Getting that run's `N`, mean ITL and `good_frac` is the
single highest-value missing measurement in the project.

## Why rewrite the control plane and not the kernels

Because the control plane is where the measured loss is.

The score is `goodput = req/s × good_frac`, where a request is *good* when its
TTFT ≤ 3000 ms **and** its mean ITL ≤ 20 ms, and the run passes at
`good_frac ≥ 0.90`. Profiling this deployment produced three facts that decide
the whole design:

| Measurement | Consequence |
|---|---|
| Sweeping concurrency on a fixed topology: ITL never exceeded 30 % of its budget while TTFT grew 10.8× and crossed 3000 ms | Requests are lost to **queueing**, not to arithmetic. A faster attention kernel cannot buy them back. |
| A good request holds a decode slot for `OSL × ITL` = 4.0 s | `req/s ≤ decode_concurrency / 4.0`. This is a hard ceiling, and the best measured run sat at ITL 17.23 ms — 14 % of the budget left unspent. |
| 25 % of prefill GPU kernel time is `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`, bandwidth bound, 190 kernels (94 layers × 2) | Per-rank all-reduce traffic scales as `(t-1)/t`, so **narrow prefill workers are faster per GPU**. That is the mechanism behind 4P1D. |

Every one of those levers — admission ordering, batch sizing, concurrency
control, topology — lives in the Python that this project replaces. None of them
live in a kernel.

## What is here

```
crates/core       SLO and goodput accounting, the calibrated capacity model, config
crates/kvcache    paged block pool, hash-chain prefix cache, router-side radix index
crates/sched      prefill ordering (Moore-Hodgson) and decode admission (AIMD on measured ITL)
crates/engine     the Engine trait, an analytic mock engine, and the TensorRT-LLM FFI seam
crates/transfer   KV transfer, including the TP-reshard mapping 4P1D depends on
crates/router     worker registry and routing priced in milliseconds of predicted TTFT
crates/frontend   OpenAI-compatible HTTP with SSE streaming
crates/worker     prefill and decode worker runtimes, plus an in-process deployment
crates/sim        deterministic discrete-event simulation of the whole deployment
crates/tuner      AIConfigurator in the loop, scored under *our* metric
crates/cli        the `trt-llm-rs` binary
```

## The three ideas worth reading the code for

**1. The prefill batch refuses to fill up when deadlines are tight.**
Bigger prefill batches are measurably faster (+11.6 % at 4–5 sequences, MoE
grouped GEMM). But every sequence in a batch gets its first token at the same
instant — batching is processor sharing, and processor sharing makes everyone
equally late. Under a per-request deadline with a 90 % threshold, three mediocre
TTFTs beat one great one and two blown ones. So the batch grows only while it
stays *deadline feasible*. With slack it fills and collects the MoE efficiency;
under pressure it collapses toward serial and collects the good requests. Same
rule, both behaviours. → `crates/sched/src/prefill.rs`

**2. Decode admission steers on measured latency, not on a fitted curve.**
We have exactly one saturated decode measurement: 53 sequences at 17.23 ms mean
ITL. That constrains `base + slope × 53 = 17.23` and nothing else — `base = 15`
and `base = 2` both fit it and disagree by 1.9× about what fits in a 20 ms
budget. So the runtime does not read a concurrency off a model. It runs AIMD
against observed step latency, plus a per-request feasibility test: a sequence
150 tokens in at 15 ms has banked slack a fresh one has not, and
`tolerable_itl = (budget × gaps_total − elapsed) / gaps_remaining` says exactly
how much. → `crates/sched/src/decode.rs`

**3. Routing is priced in milliseconds, so prefix reuse needs no tuning weight.**
`predicted_ttft = queued_tokens/rate + (prompt − prefix_hit)/rate + transfer`.
A cache-affinity *bonus* needs a weight nobody can derive and that goes wrong as
soon as queue depth changes. Counting a prefix hit as tokens you do not have to
compute is already in milliseconds. With `--cache-bust` the term is zero and
this degenerates to least-predicted-wait, which is correct for the scored run.
→ `crates/router/src/policy.rs`

## Working without a GPU

Policy work costs nothing here. The simulator runs the *real* schedulers, router
and admission rules against a cost model fitted to measured hardware, in virtual
time, deterministically.

```bash
trt-llm-rs -c configs/qwen3-4p1d.toml plan          # rank topologies
trt-llm-rs -c configs/qwen3-4p1d.toml sim           # score one config
trt-llm-rs -c configs/qwen3-4p1d.toml sweep         # policy x concurrency table
trt-llm-rs -c configs/qwen3-4p1d.toml serve         # real HTTP, mock engines
```

`serve` brings up the whole control plane over OpenAI-compatible HTTP with mock
engines, so the frontend, router, both worker runtimes and the KV bookkeeping
can be exercised end to end on a laptop.

**The simulator has no source of variance** — fixed ISL/OSL, deterministic batch
costs, no stragglers, no cold start, no fabric jitter — so it reports
`good_frac` near 1.0 where real runs land near 0.93. Absolute numbers from it
are not comparable to a measured run; *differences between policies under the
same conditions* are, and that is what it is for.

What it says today, on the calibrated 16-GPU configuration:

| prefill shape | goodput | TTFT p99 | ITL mean | mean batch | decode C |
|---|---|---|---|---|---|
| 4 × TP2 (4P1D) | 16.67 | 1224 ms | 18.02 ms | 4.01 seq | 58.3 |
| 2 × TP4 (2P2D) | 15.63 | 1248 ms | 17.45 ms | 5.07 seq | 53.3 |
| 1 × TP8        | 15.05 |  681 ms | 17.14 ms | 5.08 seq | 50.4 |

The ordering follows the all-reduce argument. Note the two rows flagged
`extrapolated_beyond_calibration`: they drove decode past 53 concurrent
sequences, which is the only point anyone has measured, so their goodput is an
extrapolation and is printed with that caveat attached.

## AIConfigurator in the loop

[AIConfigurator](https://github.com/ai-dynamo/aiconfigurator) is a good
*generator* of candidate layouts and a poor *judge* of this deployment: it
filters on TTFT/TPOT targets rather than on per-request `good_frac`, and its own
documentation says it models no scheduler, no KV behaviour and no queueing —
which is precisely the gap this project fills.

```bash
trt-llm-rs -c configs/qwen3-4p1d.toml tune                       # print the commands
aiconfigurator cli support --model-path ... --system h200_sxm --backend trtllm
aiconfigurator cli default --save-dir ./aic-results ...
trt-llm-rs -c configs/qwen3-4p1d.toml tune --save-dir ./aic-results
```

The last command reads AIConfigurator's Pareto rows, cross-checks each against
our own measured capacity model (a disagreement is information — it says which
assumption to go and measure), scores every candidate in simulation under the
real metric, and ranks them. See `docs/aiconfigurator.md`.

## Honesty rules this repository enforces on itself

- **A calibration constant is labelled with the run it came from.** Anything not
  measured says so in the doc comment, and `DecodeCurve` carries a test whose
  only purpose is to keep the unidentifiability visible.
- **A simulated result that ran past the calibrated point says so.**
  `SimReport::caveats()` is printed next to the number, every time.
- **Prefix-cache hit rate is reported with every run.** A result produced at a
  non-zero hit rate on a cache-busted workload measures the cache, not the
  model. This project has already lost a "17.67 goodput" record to exactly that.
- **A KV transfer that moved zero bytes is a failure, whatever it returned.**
  `TransferStats::moved_data()` exists so nothing downstream is interpreted
  before that is checked.

## Status, and what has actually been measured on hardware

The Rust scheduler runs live inside TensorRT-LLM on 16 GPUs. It is installed
through `scripts/usercustomize.py`, which reaches the MPI-spawned executor
ranks, rather than through the C++ Executor FFI — that seam
(`crates/engine` feature `trtllm`) and the NIXL binding (`crates/transfer`
feature `nixl`) are still unbuilt, and `docs/trtllm-ffi.md` and
`docs/kv-transfer.md` say so. The TP-reshard *mapping* NIXL depends on is
implemented and tested, because that is the part that fails silently.

**Job 316364, 4P1D, 16 GPUs.** RustScheduler live in all eight decode ranks:
599,299 steps, 0 fallbacks, 0 malformed, 0 errors, engine-side ITL p99
15.4 ms. The control plane works.

**Job 316849, the first complete official AIPerf run. Goodput 0.00 req/s.**
TTFT passes (p90 2,981 ms against a 3,000 ms gate). ITL does not: 91 ms against
20 ms. The decode engine advanced 26 of 80 resident sequences per step and the
queueing turned a 15 ms engine step into a 91 ms user-visible interval. Request
throughput 4.08 req/s, output 815 tok/s.

Three causes were found afterwards and are fixed but **not yet re-measured**,
because the GPU budget ran out:

- `--enable-cuda-graph` is accepted by `dynamo.trtllm` and wired only into its
  diffusion engine; the LLM path references it zero times. CUDA graphs were
  never on. They now go through `--extra-engine-args`, which is itself
  unvalidated upstream, so `scripts/validate_engine_yaml.py` checks the keys.
- `max_seq_len` defaulted to the model's 262,144-token window while the
  capacity policy is `GUARANTEED_NO_EVICT`, so the engine reserved against that
  rather than the 4,200 tokens this workload uses.
- One TP8 decode worker delivered 815 tok/s. Qwen3-235B has 4 KV heads: TP4
  gives each rank one, TP8 must duplicate. The topology is now 2P2D on TP4.

The predicted result of that configuration, derived rather than hoped for, is
in the `N` default of `scripts/stage-d-235b-disagg.sbatch`: N=80 is the largest
declarable concurrency that can still pass, and passing it means goodput
11.46 req/s. It rests on two numbers the next run should read off directly —
prefill sustaining 45,845 tok/s and decode delivering 2,281 tok/s.

**No run in this tree has yet passed the official criterion.** The 16.4 quoted
at the top is an external reference and remains unmatched.

## The A/B surface, and what the model says binds

Every swappable implementation is one variable in
`scripts/stage-d-235b-disagg.sbatch`, so a comparison is this stack against
itself with one thing changed. Values come from the TensorRT-LLM source;
availability from importing the module in the container.

| Variable | Values | Note |
|---|---|---|
| `ATTN_BACKEND` | `TRTLLM` (default), `FLASHINFER`, `TRITON`, `VANILLA` | flashinfer is installed |
| `MOE_BACKEND` | `AUTO` (default, resolves to CUTLASS on SM90), `DEEPGEMM`, `CUTLASS`, `TRTLLM`, … | `WIDEEP` needs `deep_ep`, absent |
| `ALLREDUCE_STRATEGY` | `AUTO` (default), `NCCL`, `ONESHOT`, `TWOSHOT`, `LOWPRECISION`, … | `MNNVL` needs a fabric this cluster does not provision |
| `XFER_BACKEND` | `UCX` (default), `NIXL`, `MPI`, `MOONCAKE` | mooncake absent |
| `EXPERT_PARALLEL` | `1` (default), or a size | see below |
| `KV_XFER_CONCURRENCY` | `16` (default) | upstream's default is **1** |
| `CHUNKED_PREFILL` | `0` (default) | a prefill worker has no decode to interleave with |

`scripts/validate_engine_yaml.py` runs before the workers start and builds the
real `TorchLlmArgs`, which catches a misspelt key, a misspelt value in a field
typed `str`, a shape the cross-field validators reject, and a correct value
whose library is not importable.

**Expert parallelism defaults to 1, and two sources agree.** Our own wiki
records SGLang job 302350 reaching goodput 14.35 at gf 0.932 with EP1 on both
sides, +43% over EP4. And TensorRT-LLM's own default is EP1: `mapping.py:112`
sets `moe_ep_size = 1` and TP-shards the experts when neither is given. Passing
`--expert-parallel-size 4` sets `moe_tp_size` to 1 — pure expert parallelism,
no MoE tensor parallelism.

**The KV transfer was serialised and nobody had noticed.**
`baseTransBuffer.cpp:109` reads
`getEnvRequestKVCacheConcurrent() ? getEnvKVCacheRecvBufferCount() : 1`, and
that flag defaults to false, so the declared buffer count of 2 is never read.
Send defaults to 1 as well. Job 316849 measured Time to Second Token at
181.85 ms with p50 191 and p99 208 — far too tight for a queue under varying
load — which is a 5.50 req/s ceiling against 4.08 measured.

### Where the remaining throughput is

`plan` prints the binding resource by name and the multiple that removing it
would buy:

```
overall prefill MFU   17.3%
  9.0% of wall time outside the forward pass    1.10x
 25.0% of kernel time in the TP all-reduce      1.33x
 the remaining 68.2% runs at 25.3% MFU          1.97x to reach 50%
```

The 25%-of-kernel-time all-reduce reads like the thing to attack and is worth
1.33x. Both cheap fixes together are 1.46x and leave the GEMMs at 25.3%. The
lever is the grouped GEMM, which is a MoE backend question, and no P/D split
changes MFU.

**`plan` and `sim` currently disagree by 2x** — 14.30 against 7.53 req/s on the
same config — and neither has been tuned to match the other. The simulator
batches 1.118 prefill sequences because it refuses to fill the batch when
deadlines are tight; the calibration behind `plan` came from a run with large
batches. Until a measurement at a known prefill batch size settles it, treat
`plan` as the optimistic bound and `sim` as the bound with that policy priced
in.

## Build

```bash
cargo test --workspace          # no GPU, no network
cargo build --release
```

On a cluster, build through the scheduler — see `scripts/build.sbatch`.
