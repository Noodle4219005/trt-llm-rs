#!/bin/bash
# Stage D: P/D disaggregated Qwen3-235B on 16 GPUs, official AIPerf command.
#
# WHY DISAGG, IN ONE LINE: aggregated ITL is f*93 + (1-f)*16.4 where f is the
# fraction of iterations carrying a prefill chunk. Separating prefill onto its
# own workers removes that term from the decode workers, which is the only way
# the model allows N >= 80 (aggregated it predicts 24.1 ms at N=80, over the
# 20 ms gate). See knowledge/experiments/2026-08-30-plan-16gpu-official-benchmark.md
#
# TOPOLOGY 2P + 2D, TP4 everywhere. Two TP4/EP4 prefill workers on node A, two
# TP4/EP4 decode workers on node B. Declared N is 80. Expected goodput if it
# passes: 11.46 req/s.
#
# TP4 AND NOT TP8, for a measured reason. Qwen3-235B has 4 KV heads: TP4 gives
# each rank exactly one, TP8 cannot divide them and duplicates -- which
# crates/transfer's own reshard tests already model
# (tp2_to_tp8_replicates_each_head_across_two_ranks). One TP8 decode worker
# measured 815 tok/s (job 316849), which caps ITL-compliant concurrency at 16
# and goodput at 4.08 req/s, no better than the 8-GPU aggregated baseline. The
# team's 2P2D on TP4 measured 2,170-2,470 tok/s on the same 8 decode GPUs.
#
# N=80 AND NOT LESS. Goodput is throughput here, and throughput rises with N
# until a gate binds, so the best declarable N is the largest that still passes.
# See the derivation at the N default below: 96 is already impossible on
# prefill, 80 is the edge, 64 is the fallback with margin at 9.17 req/s.
#
# THREE SHAPES THAT ARE RETRACTED, kept so they are not re-derived:
#  - 1P+3D, sized from a measured 3.94 req/s. Circular: that figure was itself
#    capped by the 8-GPU config it came from, and one TP4 prefill worker tops
#    out near 10.75 req/s, far above what it was sized against.
#  - 4P+1D on TP2 prefill. 220.2 GiB of FP8 weights over TP2 is 110.1 GiB on a
#    131 GiB card, and the ~21 GiB left cannot hold the activation workspace,
#    the CUDA graphs and a KV pool at once. Measured from both sides:
#    kv=0.30 serves but dies under AIPerf's opening burst with "No free block
#    found"; kv=0.50 stalls in warmup and never registers; kv=0.70 OOMs at
#    startup. The window is empty.
#  - 2P+1D with a TP8 decode worker, above.
#
# WHAT THIS DOES AND DOES NOT TEST. The P/D handoff is upstream's
# (dynamo.trtllm --disaggregation-mode + TensorRT-LLM's cache transceiver); our
# Rust frontend and router are NOT in this path. What IS ours and IS live is the
# RustScheduler policy inside every worker, armed through usercustomize. That is
# deliberate: the question this job answers is whether removing prefill
# interference keeps decode ITL near 16.4 ms at high N. If it does not, building
# a Rust P/D handoff would be wasted, so the physics is measured with the
# cheapest apparatus first.
#
# THREE TRAPS, ALL ALREADY PAID FOR:
#  1. --disaggregation-mode alone does NOT enable KV movement. The prefill worker
#     loads and registers cleanly and then dies on the FIRST REAL REQUEST with
#     "kv_cache_transceiver is disabled". cache_transceiver_config is mandatory.
#  2. backend: UCX, not DEFAULT. DEFAULT resolves to NIXL, whose agent takes a
#     FileLock on the hardcoded /tmp/trtllm_nixl_port.lock -- and /tmp is shared
#     by every job on the node.
#  3. /v1/models LIES about readiness in disagg: the frontend lists the model
#     before the P/D pair can serve, then returns 503 "not ready to serve
#     requests yet". The readiness signal is a real request returning 200.
#     (job 312838 paid 13.9 SU for 1 and 2; A2's gate paid for 3.)
#
#SBATCH -J stage-d-235b-disagg
#SBATCH -A acd110018
# Partition/nodes/GPUs come from the submit line. The defaults below are the
# 16-GPU 2P2D shape; --exclude is not optional, see the NCCL note on 143.
#   sbatch -p 16gpus -N 2 --gres=gpu:H200:8 -t 00:15:00 \
#     --exclude=25a-hgpn142,25a-hgpn143 scripts/stage-d-235b-disagg.sbatch
#   ... --export=ALL,N=64   # the fallback with margin, goodput 9.17 req/s
#SBATCH --ntasks-per-node=1
#SBATCH -c 96
#SBATCH --mem=0
#SBATCH -t 01:30:00
#SBATCH -o /work/u5727520/trt-llm-rs-runs/logs/%x-%j.out
#SBATCH -e /work/u5727520/trt-llm-rs-runs/logs/%x-%j.err

set -uo pipefail
# N=80, and it is the theoretical optimum rather than a round number.
#
# Goodput IS throughput here, and throughput rises with N until a gate binds,
# so the best declarable N is the largest one that still passes. Inverting
# Little's law -- which held to 2.5% on job 316849 -- with ITL sitting exactly
# on its 20 ms budget (that maximises the latency budget and so minimises the
# throughput required):
#
#   N    throughput   prefill needed   vs p90 50,925   decode needed   vs 2,170
#   32     4.58/s          18,338          +178%             912         +138%
#   64     9.17/s          36,676           +39%           1,825          +19%
#   80    11.46/s          45,845           +11%           2,281           -5%
#   96    13.75/s          55,014            -7%           2,737          -21%
#
# 96 is already impossible on prefill. 80 is the edge, and passing it means
# goodput 11.46 req/s.
#
# The prefill figure is the p90 of Active Prefill Throughput from job 316849,
# NOT its average. The average was 27,870 and reading that as capacity is what
# produced an earlier estimate of 7 req/s -- but prefill sat at a concurrency
# of 4.06 out of 80 in that run, so the average measures how idle it was, not
# how fast it is. p50 21,027 / p90 50,925 / p99 113,961 / max 129,939.
#
# TWO CONDITIONS DECIDE THIS RUN, and both are worth reading off the result
# rather than just the pass/fail:
#   1. prefill sustains >= 45,845 tok/s      (+11% over the p90 it has shown)
#   2. decode delivers  >= 2,281 tok/s       (SGLang 2xTP4 reference: 2,170-2,470)
# Decode is the tighter side, and it is exactly where cuda_graph_config and
# --max-seq-len 4608 should land: job 316849 ran with CUDA graphs off and a
# 262,144-token reservation, and the SGLang reference numbers come from an
# engine with neither handicap.
#
# If it fails, N=64 is the derived fallback: same configuration, +39% and +19%
# margin, goodput 9.17 req/s.
: "${N:=128}"
# The official AIPerf shape. Reported by the startup KV-pool check for context;
# the check itself sizes against MAX_SEQ_LEN, because GUARANTEED_NO_EVICT
# reserves that per admitted request rather than what the request will use.
# Keep these in step with the --synthetic-input-tokens-mean and
# --output-tokens-mean passed to aiperf below.
# ---------------------------------------------------------------------------
# One source of truth for every knob below.
#
# `trt-llm-rs config --emit-env` turns the TOML into this file, and a test
# asserts that every `: "${X:=` in this script appears in it and vice versa --
# so a knob cannot exist on one side only. Sourcing it here rather than
# assigning means the layering stays what a person expects: a variable set on
# the sbatch command line beats the file, which beats the defaults below.
#
# Point TRTLLM_RS_ENV at your own file to tune without editing this script.
# Defined here rather than with the other paths below, because the knobs are
# declared above those and this has to exist before they are sourced.
SRC="${SRC:-/home/u5727520/trt-llm-rs}"
DEPLOY_ENV="${TRTLLM_RS_ENV:-$SRC/deployment.env}"
if [ -f "$DEPLOY_ENV" ]; then
  echo "config: sourcing $DEPLOY_ENV"
  # shellcheck disable=SC1090
  . "$DEPLOY_ENV"
else
  echo "config: no $DEPLOY_ENV, using the defaults compiled into this script"
fi

: "${ISL:=4000}"
: "${OSL:=200}"
# Two prefill workers of four GPUs each, not four of two. Four TP2 workers is
# the shape the team measures highest on, and it is not reachable on this
# hardware with TensorRT-LLM: 220.2 GiB of
# FP8 weights over TP2 is 110.1 GiB on a 131 GiB card, and the ~21 GiB left has
# to hold the activation workspace AND the KV pool. Three measurements close the
# window from both sides:
#
#   PREFILL_KV_FRACTION=0.30  starts, then dies under AIPerf's opening burst
#                             with "No free block found"          (316364)
#   PREFILL_KV_FRACTION=0.70  cudaMalloc OOM at startup, t=54s    (316414)
#   PREFILL_KV_FRACTION=0.80  cudaMalloc OOM at startup           (316051, 316287)
#
# Four GPUs per prefill worker removes the constraint rather than balancing on
# it: 55 GiB of weights per rank leaves ~76 GiB, so 0.70 is ~53 GiB of KV
# against ~23 GiB of headroom. Two prefill workers also have the throughput --
# one tops out near 10.75 req/s at 93 ms per 4000-token prompt, so two is
# ~21.5 req/s, above the 19.5 the team has measured end to end. Still 16 GPUs:
# 2x4 on the prefill node, 2x4 on the decode node.
#
# The weight arithmetic does NOT depend on which axis the four are split on --
# 220.2 GiB over four GPUs is 55 GiB each however it is cut. What the axis
# changes is the activation workspace, which is what made 0.70 OOM at two GPUs
# per worker. PP2 x DP2 was the requested way to probe that and did not survive
# NCCL init; see the PREFILL_PP note above.
: "${PREFILL_WORKERS:=4}"
: "${PREFILL_TP:=2}"
# PP2 x attention-DP over the same four GPUs was tried (316457) and never got as
# far as allocating memory: NCCL communicator probe initialization timed out
# after 60 s on all four ranks, in both workers, while the TP8 decode worker on
# the other node came up normally in the same job. gpus_per_node equalled world
# size in every case, so the device-isolation confusion recorded earlier is not
# the explanation and the cause is still unknown. Plain TP keeps the NCCL
# topology the same shape as the TP2 and TP8 workers that do work.
#   PREFILL_PP=2 PREFILL_ATTENTION_DP=1  to reproduce it.
# --enable-attention-dp has no separate size flag: attention_dp_size equals
# tensor_parallel_size.
: "${PREFILL_PP:=1}"
: "${PREFILL_ATTENTION_DP:=0}"
# Decode is 2xTP4, not 1xTP8. Job 316849 measured 815 tok/s out of one TP8
# decode worker, which caps ITL-compliant concurrency at 815 x 0.020 = 16 and
# goodput at 4.08 req/s -- no better than the 8-GPU aggregated baseline. The
# team's 2P2D with TP4 decode measured 12.35 req/s and avg ITL 19.81 ms on the
# same 8 decode GPUs. Qwen3-235B has 4 KV heads: TP4 gives each rank exactly
# one, TP8 cannot divide them and has to duplicate.
: "${DECODE_WORKERS:=2}"
: "${DECODE_TP:=4}"

# The engine defaults to the model's full 262,144-token window, and the
# capacity policy is GUARANTEED_NO_EVICT -- it only admits what it can promise
# never to evict, so it reserves against that window instead of the 4,200
# tokens this workload uses. ISL 4000 + OSL 200 + margin.
: "${MAX_SEQ_LEN:=4608}"

# HOW MANY KV TRANSFERS MAY BE IN FLIGHT. Upstream's default is ONE, in each
# direction, and it is not written as a default -- it is written as a branch:
#
#   baseTransBuffer.cpp:109
#     mRecvBufferCount = getEnvRequestKVCacheConcurrent() ? getEnvKVCacheRecvBufferCount() : 1;
#   baseTransBuffer.cpp:110
#     mSendBufferCount = getEnvKVCacheSendMaxConcurrenceNum();
#
#   TRTLLM_REQUEST_KV_CACHE_CONCURRENT       envUtils.cpp:382   default false
#   TRTLLM_KVCACHE_RECV_BUFFER_COUNT         envUtils.cpp:476   default 2, gated by the above
#   TRTLLM_KVCACHE_SEND_MAX_CONCURRENCY_NUM  envUtils.cpp:470   default 1
#
# So both pools hold one buffer, and assignBufferIndex blocks on a condition
# variable when it is taken (baseTransBuffer.h:231-238). The whole P/D handoff
# is one transfer at a time.
#
# This is what job 316849 measured without recognising it. Time to Second Token
# -- the gap between the prefill worker's first token and the decode worker's
# second -- was avg 181.85 ms with p50 191, p90 195, p99 208: a distribution
# far too tight for a queue under varying load, and exactly what a fixed
# service time looks like. 1000/181.85 = 5.50 req/s is the ceiling that implies,
# against 4.08 req/s measured. Meanwhile AIPerf counted 73.95 requests in the
# decode phase while the engine offered 26.3 candidates per step: the missing
# 48 were queued behind a single buffer.
#
# The memory was already paid for. max_tokens_in_buffer was 65536 -- sixteen
# 4000-token requests, about 1.5 GB per rank -- sized into a pool of one, so
# fifteen requests' worth of buffer sat idle while fifteen requests waited.
# One request per buffer and sixteen buffers is the same footprint and sixteen
# times the concurrency. 4096 rather than 4000 because
# computeTransferBufferSize asserts maxNumTokens % tokensPerBlock == 0
# (cacheTransBuffer.cpp:212) and tokensPerBlock is 32.
#
# vLLM makes the opposite choice by construction: its connector reports
# completion as a SET of request ids (get_finished) and its scheduler tracks
# finished_recving_kv_req_ids as a set, so many transfers in flight is the
# normal case and there is no single buffer to serialise on.
: "${KV_XFER_CONCURRENCY:=16}"

# ---------------------------------------------------------------------------
# BACKEND A/B SURFACE. One variable per swappable implementation, so a
# comparison is "this stack with backend X against the same stack with backend
# Y" and never "framework A against library B" -- the operator layer only
# admits the first kind of claim.
#
# Values are the Literals TensorRT-LLM v1.3.0rc22 actually accepts, read from
# the source rather than from a compatibility table:
#
#   attn_backend        TRTLLM (default) | FLASHINFER | TRITON | VANILLA
#                       attention_backend/{trtllm,flashinfer,triton_prefill,vanilla}.py
#   moe_config.backend  AUTO (default) | CUTLASS | CUTEDSL | WIDEEP | TRTLLM |
#                       DEEPGEMM | DENSEGEMM | VANILLA | TRITON | MARLIN |
#                       MEGAMOE_DEEPGEMM                     llm_args.py MoeConfig
#
#   AUTO IS CUTLASS HERE, and that is worth knowing before reading a result as
#   "the default". model_config.py:363-369 returns TRTLLM only for
#   fp8_block_scales on SM100 (Blackwell); an H200 is SM90, so the branch cannot
#   fire and it falls through to CUTLASS.
#
#   DEEPGEMM is available and was gated by mistake until 2026-09-01. It does
#   NOT need the PyPI `deep_gemm`: fused_moe_deepgemm.py imports
#   `from tensorrt_llm import deep_gemm`, a bundled module, and every symbol
#   that path uses is present. SGLang's SOTA run set its own MoE backend to
#   deep_gemm explicitly -- suggestive, but a DIFFERENT implementation, so it
#   is a reason to try DEEPGEMM here, not evidence about it. The EP1 finding
#   above is mechanism-level and does transfer; this one is not.
#   transceiver backend DEFAULT | UCX | NIXL | MOONCAKE | MPI  llm_args.py:3836
#
# WHAT IS ACTUALLY IMPORTABLE IN THIS CONTAINER, checked, not assumed:
#   flashinfer  yes      nixl  yes      kvbm  yes
#   tensorrt_llm.deep_gemm  yes          deep_ep  NO
#   mooncake    NO       lmcache  NO
# So WIDEEP fails at import (fused_moe_wide_ep.py wants deep_ep), MOONCAKE has
# no library behind it, and the kv_connector registry's lmcache entries point
# at a module that is not here (connectors/registry.py:23-31). kvbm is the one
# connector that could be switched on today, and it is the wrong one to switch
# on: its API is get_num_new_matched_tokens plus offload filtering, and
# --cache-bust first_turn_prefix makes the first return zero.
#
# Qwen3-235B-A22B is a MoE model, so moe_config.backend is the larger operator
# lever of the two; attn_backend is the one with a working alternative
# installed.
: "${ATTN_BACKEND:=TRTLLM}"
: "${MOE_BACKEND:=AUTO}"
: "${XFER_BACKEND:=UCX}"

# The fourth axis, and the one the MFU breakdown points at second. The TP
# all-reduce is 25% of prefill kernel time, so removing it entirely would be
# 1.33x -- less than the 1.97x in the grouped GEMM, but the cheapest of the
# two to try because it is a one-word change.
#
#   AUTO (default) | NCCL | UB | MINLATENCY | ONESHOT | TWOSHOT |
#   LOWPRECISION | MNNVL | NCCL_SYMMETRIC
#
# MNNVL wants an NVLink fabric this cluster does not provision: every worker
# logs "the NVLink fabric/IMEX plane is NOT provisioned on this node, so NVLS
# multicast memory cannot be bound". Do not spend a run on it without fixing
# that first.
#
# NOT tried, and the reason is in the field description rather than a guess:
# enable_min_latency is documented "Currently only used for Llama4".
: "${ALLREDUCE_STRATEGY:=AUTO}"

# EXPERT PARALLELISM. EP1, and two independent sources say so.
#
# This script passed --expert-parallel-size $tp, so EP4, until both of these
# turned up:
#
# 1. Our own measurement. SGLang job 302350 on the same model and hardware
#    reached goodput 14.35 at gf 0.932 with EP1 on both sides -- dropping the
#    per-token all-to-all -- and the note records the difference from EP4 as
#    +43%. It has been in this wiki since 2026-08-26.
#
# 2. TensorRT-LLM's own default. mapping.py:112-114:
#
#        if moe_tp_size == -1 and moe_ep_size == -1:
#            moe_tp_size = moe_world_size // moe_cluster_size
#            moe_ep_size = 1
#
#    Passing nothing gives TP-sharded MoE at EP1. Passing --expert-parallel-size 4
#    sets moe_ep_size = 4, and moe_tp_size then falls out as
#    moe_world_size // (4 * 1) = 1: pure expert parallelism with no MoE tensor
#    parallelism at all. We were overriding the default to reach the slower one.
#
# EP1 with TP4 does not replicate 128 experts per rank. The expert weights are
# TP-sharded rather than EP-sharded: same memory, an all-reduce instead of an
# all-to-all. Set EXPERT_PARALLEL=4 to go back and measure the 43% for
# ourselves on this stack rather than inheriting it from SGLang's.
: "${EXPERT_PARALLEL:=1}"

# CONTEXT PARALLELISM. Off, and the arithmetic says it should not be.
#
# TTFT is the gate that binds at the best configuration anyone has measured:
# SGLang job 302350 had ITL p99 19.09 inside a 20 ms budget and TTFT p95 3526
# against 3000, failing ~7%. TTFT there is dominated by the prefill forward
# pass, 1.6-2.1 s for 4000 tokens. Context parallelism splits the sequence
# rather than the weights, so it attacks that number directly -- it lowers
# prefill LATENCY, which is the quantity the gate is written in, rather than
# throughput.
#
#   Ulysses CP2 costs two all-to-alls per layer: 4000 x 4096 x 2 B = 33 MB
#   each, x 2 x 94 layers = 6.2 GB per request. Over intra-node NVLink P2P
#   that is 31-123 ms depending on achieved bandwidth, against 800-1050 ms
#   saved by halving a 1.6-2.1 s forward. Roughly 8x net at the pessimistic end.
#
# MEMORY: CP does NOT shard weights. TP4 x CP2 needs 8 GPUs per prefill worker
# and leaves 76 GiB per rank, which is fine; TP2 x CP2 on 4 GPUs leaves 21 GiB
# and is the configuration already measured unrunnable.
#
# WHY IT IS STILL OFF: every mention of CP in TensorRT-LLM's docs is helix
# (million-token DECODE with KV sharding) or video generation. Qwen3Attention
# carries cp_size awareness (modeling_qwen3.py:138) and the MoE variant imports
# it, so the path exists, but nothing in this version documents Ulysses CP for
# a 4000-token prefill. The arithmetic is a reason to test it, not evidence
# that it works.
#
# CONTEXT_PARALLEL=2 with PREFILL_WORKERS=1 and PREFILL_TP=4 is the shape to try.
: "${CONTEXT_PARALLEL:=1}"

# ---------------------------------------------------------------------------
# ONE KNOB PER MFU LOSS TERM. `plan` reports where the tensor cores go; these
# are what can be done about each line of it.
#
#   9.0% of wall time outside the forward pass   1.10x   HOST_DISPATCH_SPIN
#  25.0% of kernel time in the TP all-reduce     1.33x   ALLREDUCE_STRATEGY (above)
#  the rest runs at 25.3% MFU                    1.97x   TORCH_COMPILE
#
# enable_low_latency_host_dispatch: "Use low-latency spin-wait mode for CUDA
# host task dispatch (cudaLaunchHostFunc_v2 with cudaHostTaskSpinWait). Reduces
# callback latency at the cost of a CPU core spinning while waiting." We
# allocate 96 CPUs per node and use a handful, so the cost is free here and the
# benefit lands exactly on the duty-cycle term.
: "${HOST_DISPATCH_SPIN:=0}"

# torch.compile. Off upstream, and the sub-options say why it is worth trying
# against the compute-MFU term rather than the others:
#   enable_fullgraph            True   (default when compile is on)
#   enable_inductor             False  <- the codegen that would change the GEMMs
#   enable_piecewise_cuda_graph False
#   enable_userbuffers          True   (same mechanism as ALLREDUCE_STRATEGY=UB)
#   max_num_streams             1      <- no overlap of independent work
#
# TORCH_COMPILE=1 enables it with upstream's defaults; TORCH_COMPILE=inductor
# additionally turns on the inductor backend, which is the one aimed at the
# 25.3%. Compilation costs startup time, which this benchmark charges to TTFT
# on the first requests, so a win has to be big enough to pay that back.
: "${TORCH_COMPILE:=0}"

# ---------------------------------------------------------------------------
# WHAT A NEIGHBOURING STACK MEASURED AT GOODPUT 22.419 ON THIS HARDWARE.
#
# Our own best valid cold-cache number is 14.35. The gap is these four, and
# each is here with the evidence that put it here rather than as a guess.

# fp8_e4m3 KV, both sides. This is what reopened TP2 prefill: our own TP2 died
# at KV fractions 0.30, 0.50 and 0.70, and every one of those ran KV in fp16.
# 220.2 GiB of weights over TP2 leaves 20.9 GiB per rank, which holds prefill's
# in-flight KV comfortably in fp8 and not in fp16. The topology was never what
# was wrong. KvCacheConfig.dtype: 'auto' inherits the checkpoint's own
# quantisation metadata, 'fp8' overrides it.
: "${KV_CACHE_DTYPE:=fp8}"

# Prefix caching OFF, and this one is about validity rather than speed.
# enable_block_reuse defaults to TRUE. A teammate's 17.67 was withdrawn on
# audit for exactly this: cached-token 93,632,679 against new-token 824,424, a
# 99.1% hit rate, prefill doing 1% of the work the benchmark meant to measure.
# AIPerf's --cache-bust first_turn_prefix should defeat it, but "should" is not
# a measurement and a headline number that does not survive an audit is worth
# less than no number.
: "${BLOCK_REUSE:=false}"

# Prefill's in-flight token budget. Upstream defaults to 8192; the 22.419
# configuration runs 16384, four whole ISL-4000 prompts in one iteration. This
# is also the quantity our own two models disagree about: `plan` calibrated on
# large batches and `sim` assumes 1.118 sequences per prefill batch, which is
# most of the 14.30-versus-7.53 gap.
: "${PREFILL_MAX_NUM_TOKENS:=16384}"

# EAGLE3 speculative decoding, decode side only.
#
# The neighbouring stack measured, on this model and this hardware: acceptance
# alpha 1.82 (median, n=33), 12/12 outputs token-identical under greedy
# sampling, and ITL p95 11.08 ms against 19.18 baseline at topk=1. topk=2 gave
# 14.97 ms and cost 5% of goodput; topk=4 collapsed to 3.29 at a 25.1% pass
# rate. One draft token is not a timid setting, it is the only measured win.
#
# The draft checkpoint is LlamaForCausalLMEagle3 with aux hidden states from
# layers 1/46/90, which is the architecture TRT-LLM's Eagle3DecodingConfig
# expects (eagle3_model_arch defaults to 'llama3').
#
# Default OFF: this is transferred evidence from a different serving stack, and
# whether TRT-LLM's Eagle3 path reproduces the ratio is what a run settles.
: "${SPEC_DECODE:=0}"
: "${SPEC_MODEL:=/work/u4063814/hf_cache/eagle3/qwen3-235b-instruct-2507-specforge}"
: "${SPEC_DRAFT_TOKENS:=1}"

# Off upstream, and left off deliberately. Chunked prefill exists to interleave
# a long prompt with decode work in the same iteration, and a disaggregated
# prefill worker has no decode work to interleave with. At ISL 4000 against
# max_num_tokens 8192 two whole prompts already fit in one iteration, which is
# what the simulator's mean_prefill_batch_tokens of ~6000-7700 shows. Set it to
# 1 to A/B that reasoning rather than trust it.
: "${CHUNKED_PREFILL:=0}"
: "${MAX_TOKENS_IN_BUFFER:=4096}"
# CUDA graphs are configured through --extra-engine-args, NOT through
# --enable-cuda-graph. That flag exists on `dynamo.trtllm`, is accepted, and is
# wired only into the diffusion engine: `grep -c enable_cuda_graph` over the LLM
# worker path returns 0, and job 316849 duly logged `cuda_graph=False` while the
# flag was being passed. A flag that is accepted and does nothing is worse than
# one that errors.
#
# Both numbers matter. CudaGraphConfig defaults to max_batch_size=0 and
# enable_padding=False, so enabling capture without setting them captures
# nothing, and without padding a decode batch that varies misses whatever was
# captured. Decode residency at N=80 is ~23 per worker; 64 leaves headroom.
: "${CUDA_GRAPH_MAX_BATCH:=64}"
# One worker per NODE, using all 8 of its GPUs. Not a preference -- job 315810
# put a prefill and a decode worker on one node with
# CUDA_VISIBLE_DEVICES=0,1,2,3 and 4,5,6,7, and the KV transfer deadlocked:
# every 60 s the decode worker logged "Generation KV cache transfer ... exceeded
# configured timeout (observe-only)" while the prefill worker logged "request
# completed". UCX itself was fine -- both sides resolved the same node IP and
# bound distinct ZMQ endpoints with no errors -- but the engine args showed
# 'gpus_per_node': 4, because the isolation made each process believe the node
# had four GPUs. TensorRT-LLM maps ranks to devices through gpus_per_node, so two
# four-GPU worlds on one eight-GPU node give the transceiver the wrong peer
# device. One worker per node removes the variable rather than guessing which
# env var works around it, and sends the transfer over IB, which is where it has
# to work for 16 GPUs anyway.

SIF=/work/u5727520/dynamo-qwen-smoke/images/tensorrtllm-runtime-1.4.1.sif
MODEL=/work/u4063814/hf_cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969
BIND="--bind /work/u4063814,/work/u5727520,/home/u5727520,$HOME/.cargo,$HOME/.rustup"
AIPERF=/home/u5727520/hpcai26-qwen/client-venv/bin/aiperf
OUT=/work/u5727520/trt-llm-rs-runs/out/stage-d-$SLURM_JOB_ID
TARGET=/work/u5727520/.trtllm-rs-target-container
STAGE="$OUT/pymod"
mkdir -p "$OUT" "$STAGE"

# One worker per NODE, one node per worker. This is the only shape that dodges
# both failure modes at once: two workers on one node gave each process a false
# gpus_per_node and deadlocked the transceiver (315810), while eight ranks on one
# node (TP8) exhausted mmap and threads during load (315919, 315920). Two nodes
# of eight GPUs, split 2x4 per node, gives four ranks per worker.
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE_A="${NODES[0]}"
HTTP_PORT=$((20000 + (SLURM_JOB_ID * 7) % 20000))

echo "=== job $SLURM_JOB_ID  nodes ${NODES[*]}  N=$N  ${PREFILL_WORKERS}P(TP$PREFILL_TP)+${DECODE_WORKERS}D(TP$DECODE_TP)  TP=EP  $(date -Is) ==="

# Per-job discovery root: the file backend defaults to /tmp/dynamo_store_kv,
# which is shared between jobs on a node and has no CLI flag.
export DYN_FILE_KV="$OUT/discovery"; mkdir -p "$DYN_FILE_KV"
unset NATS_SERVER ETCD_ENDPOINTS ETCD_URL 2>/dev/null
export DYN_LOG=info

if [ "${CHUNKED_PREFILL}" = 1 ]; then CHUNKED_PREFILL_BOOL=true; else CHUNKED_PREFILL_BOOL=false; fi

# TWO FILES, because the two roles do not run the same phases.
#
# A disaggregated prefill worker never decodes, so a decode-mode CUDA graph on
# it captures a phase it will not execute and charges the memory anyway. The
# other direction is measured: SGLang's EP1 run had to pass
# --disable-prefill-cuda-graph because the prefill graphs wanted 9.25 GiB and
# OOMed, and the recorded fix was lowering the prefill memory fraction rather
# than keeping them.
#
# Everything else -- the backends, the transceiver, chunked prefill -- is
# shared, so it is written once into each file from the same variables.
write_engine_yaml () {
  local path="$1" role="$2"
  {
    echo "attn_backend: ${ATTN_BACKEND}"
    if [ "${CONTEXT_PARALLEL}" != 1 ]; then
      echo "context_parallel_size: ${CONTEXT_PARALLEL}"
    fi
    if [ "${HOST_DISPATCH_SPIN}" = 1 ]; then
      echo "enable_low_latency_host_dispatch: true"
    fi
    # KV cache: dtype and block reuse. Both sides.
    echo "kv_cache_config:"
    echo "  dtype: ${KV_CACHE_DTYPE}"
    echo "  enable_block_reuse: ${BLOCK_REUSE}"
    if [ "$role" = prefill ]; then
      # Prefill's in-flight budget. Decode leaves max_num_tokens alone: its
      # iterations carry one token per sequence, not whole prompts.
      echo "max_num_tokens: ${PREFILL_MAX_NUM_TOKENS}"
    elif [ "${SPEC_DECODE}" != 0 ]; then
      echo "speculative_config:"
      echo "  decoding_type: Eagle3"
      echo "  speculative_model: ${SPEC_MODEL}"
      echo "  max_draft_len: ${SPEC_DRAFT_TOKENS}"
      echo "  eagle3_one_model: true"
      echo "  greedy_sampling: true"
    fi
    case "${TORCH_COMPILE}" in
      0) ;;
      inductor)
        echo "torch_compile_config:"
        echo "  enable_inductor: true"
        ;;
      *)
        echo "torch_compile_config: {}"
        ;;
    esac
    echo "allreduce_strategy: ${ALLREDUCE_STRATEGY}"
    echo "enable_chunked_prefill: ${CHUNKED_PREFILL_BOOL}"
    echo "moe_config:"
    echo "  backend: ${MOE_BACKEND}"
    if [ "$role" = decode ]; then
      echo "cuda_graph_config:"
      # Explicit, though not required. cuda_graph_config is a union
      # discriminated on `mode`, and validating the annotation directly rejects
      # a dict without it -- but the engine does not validate the annotation
      # directly: TorchLlmArgs.infer_cuda_graph_config_mode fills `mode` in
      # first, so it resolves to DecodeCudaGraphConfig either way. An earlier
      # version of this comment claimed omitting it kills the worker. It does
      # not; the claim came from checking the wrong surface.
      echo "  mode: decode"
      # max_batch_size defaults to 0 and enable_padding to False, so enabling
      # capture without setting them captures nothing, and without padding a
      # varying decode batch misses what little it did capture.
      # Speculation multiplies the batch dimension: a step for `cap`
      # sequences runs cap x (1 + draft) rows. A capture sized for cap alone
      # silently misses every speculative step -- the same class of failure as
      # max_batch_size 0, which is configured, enabled, and does nothing.
      graph_cap="${CUDA_GRAPH_MAX_BATCH}"
      if [ "$role" = decode ] && [ "${SPEC_DECODE}" != 0 ]; then
        graph_cap=$(( CUDA_GRAPH_MAX_BATCH * (1 + SPEC_DRAFT_TOKENS) ))
      fi
      echo "  max_batch_size: ${graph_cap}"
      echo "  enable_padding: true"
    fi
    echo "cache_transceiver_config:"
    # UCX rather than DEFAULT: DEFAULT resolves to NIXL, which FileLocks a
    # hardcoded path under the node-shared /tmp.
    echo "  backend: ${XFER_BACKEND}"
    # One request per buffer; the buffer COUNT is what carries concurrency and
    # it lives in TRTLLM_KVCACHE_*, not here. See KV_XFER_CONCURRENCY above.
    echo "  max_tokens_in_buffer: ${MAX_TOKENS_IN_BUFFER}"
  } > "$path"
}
write_engine_yaml "$OUT/engine-prefill.yaml" prefill
write_engine_yaml "$OUT/engine-decode.yaml" decode
for role in prefill decode; do
  echo "  engine-$role.yaml:"; sed 's/^/    /' "$OUT/engine-$role.yaml"
done

PIDS=()
cleanup () {
  echo; echo "echo; echo "########## verdict ##########"
# The run says whether it met the prediction it printed at the top, and when it
# did not, which gate held and which broke. Job 316849 needed a person reading a
# console table to work out that TTFT passed and ITL missed by 4.6x.
EXPORT="$OUT/aiperf/profile_export_aiperf.json"
if [ -f "$EXPORT" ]; then
  srun -N1 -n1 -w "$NODE_A" --overlap apptainer exec $BIND "$SIF" \
    "$STAGE/trt-llm-rs" verdict --aiperf "$EXPORT" 2>&1 | sed 's/^/  /' \
    || echo "  (verdict unavailable)"
else
  echo "  no AIPerf export at $EXPORT -- nothing to assess"
fi

########## teardown ##########"
  kill "${PIDS[@]}" 2>/dev/null; sleep 10
  srun -N2 -n2 --overlap bash -c 'pkill -f "dynamo.trtllm"; pkill -f "dynamo.frontend"' 2>/dev/null
  sleep 5
}
trap cleanup EXIT

########## build the Rust policy module once ##########
echo; echo "########## build trtllm_rs ##########"
srun -N1 -n1 -w "$NODE_A" --overlap apptainer exec --nv $BIND \
  --env "CARGO_TARGET_DIR=$TARGET" --env "CARGO_BUILD_JOBS=16" \
  --env "PYO3_BUILD_EXTENSION_MODULE=1" --env "SRC=$SRC" --env "STAGE=$STAGE" \
  --env "CARGO_NET_OFFLINE=true" \
  "$SIF" bash -euo pipefail -c '
    export PATH="$HOME/.cargo/bin:$PATH"; cd "$SRC"
    cargo build --release -p trtllm-pyext
    cp "$CARGO_TARGET_DIR/release/libtrtllm_pyext.so" "$STAGE/trtllm_rs.so"
    # The planner too, so this run records what was predicted for it before it
    # produces what actually happened. Six jobs were interpreted against a model
    # that had never been printed next to a result.
    cargo build --release -p trtllm-cli
    cp "$CARGO_TARGET_DIR/release/trt-llm-rs" "$STAGE/trt-llm-rs"' || exit 20
cp "$SRC/scripts/nccl_preflight.py" \
   "$SRC/scripts/usercustomize.py" "$SRC/scripts/trtllm_rs_shadow.py" \
   "$SRC/scripts/trtllm_rs_mmap_weights.py" "$STAGE/"

########## frontend on node A ##########
echo; echo "########## frontend on $NODE_A:$HTTP_PORT ##########"
srun -N1 -n1 -w "$NODE_A" --overlap apptainer exec --nv $BIND "$SIF" \
  python3 -m dynamo.frontend --http-port "$HTTP_PORT" \
    --discovery-backend file --event-plane zmq \
    > "$OUT/frontend.log" 2>&1 &
PIDS+=($!)
sleep 20

########## workers: decode first, then prefill in parallel ##########
# Every TP rank mmaps the whole 221 GB checkpoint; two workers starting together
# on one node exhaust the cgroup (job 314880: "unable to mmap ... Cannot allocate
# memory" plus "can't start new thread"). Serialising is nearly free because the
# first worker pulls the weights into page cache for the ones after it.
# 2P2D with TP == EP. Two TP4/EP4 prefill workers share node A and two TP4/EP4
# decode workers share node B, four GPUs each, so both sides need
# CUDA_VISIBLE_DEVICES isolation.
#
# The device isolation on node A is NOT believed to be the deadlock cause any
# more: job 315954 removed it entirely -- one worker per node, gpus_per_node 8,
# cross-node IB, TP4 -- and deadlocked identically. The untested fix is
# max_tokens_in_buffer, set in engine.yaml above; job 316005 was SIGTERMed by
# something external at t=3:10, before AIPerf ever ran, so it has still never
# been exercised.
# The GPUs of the node this worker is about to use must actually be free.
# Job 316166 put the decode worker first and it died on rank 1 with
# `cudaErrorDevicesUnavailable` at torch.cuda.set_device -- before weight
# loading, at a step every earlier run had passed. The earlier runs started
# prefill first, so node B was not touched for eight minutes; that delay was
# never a design decision, it just happened to outlast whatever still held the
# devices. Waiting on the observable condition is the fix; re-adding the
# accidental delay is not.
# A node whose NCCL is broken costs a full run if it is found after the weights
# are loaded rather than before. 25a-hgpn143 hangs on the first all_reduce with
# init_process_group reporting success, which cost 316457, 316497 and 316571
# about 510 SU between them and produced three different wrong explanations.
nccl_preflight () {
  local node="$1" n="$2"
  if ! timeout 120 srun -N1 -n1 -w "$node" --overlap apptainer exec --nv $BIND \
        --env "PYTHONPATH=$STAGE" "$SIF" \
        torchrun --nproc_per_node="$n" --nnodes=1 \
        --master_port=$(( 21000 + SLURM_JOB_ID % 9000 )) \
        "$STAGE/nccl_preflight.py" > "$OUT/nccl-$node.log" 2>&1; then
    echo "!!! NCCL PREFLIGHT FAILED on $node -- aborting before loading 220 GiB"
    echo "!!!   resubmit with --exclude=$node ; see $OUT/nccl-$node.log"
    tail -15 "$OUT/nccl-$node.log"
    exit 27
  fi
  grep -h "NCCL ok" "$OUT/nccl-$node.log"
}

GATED_NODES=""
await_free_gpus () {
  local node="$1" i busy
  # Once per node only. After our own first worker starts, its ranks ARE the
  # compute processes, so gating every worker would make each subsequent one
  # wait out the full 100 s against ourselves -- three times on the prefill
  # node, five minutes, 80 SU, to observe a condition we created.
  case " $GATED_NODES " in *" $node "*) return 0 ;; esac
  GATED_NODES="$GATED_NODES $node"
  for i in $(seq 1 20); do
    # `grep -c .` prints 0 and EXITS 1 when it counts nothing, so a `|| echo 0`
    # here appends a second zero, "$busy" becomes "0\n0", and the test below can
    # never match: job 316287 waited the full 100 s on BOTH nodes, ~53 SU, to
    # observe that nothing was running.
    busy=$(srun -N1 -n1 -w "$node" --overlap nvidia-smi \
             --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    [ "$busy" = "0" ] && { [ "$i" -gt 1 ] && echo "    $node GPUs free after $((i*5))s"; return 0; }
    sleep 5
  done
  echo "!!! $node still has $busy compute processes after 100s; starting anyway"
}

# Launch and wait are separate so the prefill workers can load at the same time.
declare -A WPID

launch_worker () {
  local role="$1" node="$2" tag="$3" tp="$4" gpus="${5:-}" pp="${6:-1}" kvfrac adp=()
  local ep="${EXPERT_PARALLEL}"
  # 220.2 GiB of FP8 weights over TP2 is 110.1 GiB on a 131 GiB card, leaving
  # ~21 GiB; asking for 80% of that on top of activations is what killed all
  # four prefill workers in 316051 and 316287. A prefill worker computes KV and
  # ships it, so its pool only has to hold the requests in flight.
  #
  # The failure at 0.30 is FATAL, not backpressure: TensorRT-LLM answers "No free
  # block found" by terminating the event loop and killing the rank (316364).
  #
  # But 0.30 was NOT short by the obvious arithmetic. Each rank logs its pool
  # twice -- a provisional allocation and then the real one -- and the real one
  # was 91,072 tokens, MORE than the 80,000 that N=80 over four workers naively
  # needs. So the naive figure does not predict this failure. Two things it
  # misses: AIPerf opens with concurrency_burst, delivering all of N at once to
  # a router that does not split it evenly, and a prefill worker cannot free a
  # block until its KV has reached the decode side.
  #
  # 0.70 buys ~212,000 tokens, about 53 requests per worker against 20 under
  # even routing. That is headroom for imbalance, not a computed fit, because
  # what to fit to is not known. 0.80 OOMs at startup (316051, 316287), so this
  # is the last stop before that edge. UNTESTED.
  #
  # If it still dies, the window is too narrow to stand on and the structural
  # answer is TP4 prefill (2P1D, still 16 GPUs): 55 GiB of weights per rank
  # leaves ~76 GiB and the pool stops being the binding constraint at all.
  # The decode worker is TP8 -- 27.5 GiB of weights, ~104 GiB free -- and keeps
  # the large pool, which is the whole point of splitting them.
  if [ "$role" = prefill ]; then kvfrac="${PREFILL_KV_FRACTION:-0.70}"
  else                           kvfrac="${DECODE_KV_FRACTION:-0.80}"; fi
  await_free_gpus "$node"
  if [ "$role" = prefill ] && [ "${PREFILL_ATTENTION_DP}" = 1 ]; then
    adp=(--enable-attention-dp)
  fi
  echo "  starting $tag: role=$role node=$node tp=$tp pp=$pp ep=$ep kv=$kvfrac adp=${#adp[@]} gpus=${gpus:-all}"
  srun -N1 -n1 -w "$node" --overlap apptainer exec --nv $BIND \
    ${gpus:+--env "CUDA_VISIBLE_DEVICES=$gpus"} \
    --env "PYTHONPATH=$STAGE" \
    --env "TRTLLM_RS_SHADOW=1" --env "TRTLLM_RS_LIVE=1" \
    --env "TRTLLM_RS_MMAP_WEIGHTS=1" \
    --env "TRTLLM_REQUEST_KV_CACHE_CONCURRENT=1" \
    --env "TRTLLM_KVCACHE_RECV_BUFFER_COUNT=${KV_XFER_CONCURRENCY}" \
    --env "TRTLLM_KVCACHE_SEND_MAX_CONCURRENCY_NUM=${KV_XFER_CONCURRENCY}" \
    --env "TRTLLM_RS_SHADOW_OUT=$OUT/bridge-$tag" \
    --env "TRTLLM_RS_ITL_BUDGET_MS=20.0" \
    --env "DYN_FILE_KV=$DYN_FILE_KV" \
    "$SIF" \
    mpirun --bind-to none --oversubscribe -H "$node:96" -n 1 \
    python3 -m dynamo.trtllm \
      --model "$MODEL" --served-model-name Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
      --tensor-parallel-size "$tp" --pipeline-parallel-size "$pp" \
      --expert-parallel-size "$ep" ${adp[@]+"${adp[@]}"} \
      --free-gpu-memory-fraction "$kvfrac" \
      --max-seq-len "$MAX_SEQ_LEN" \
      --disaggregation-mode "$role" \
      --extra-engine-args "$OUT/engine-$role.yaml" \
      --discovery-backend file --event-plane zmq \
      > "$OUT/$tag.log" 2>&1 &
  WPID[$tag]=$!
  PIDS+=($!)
}

# Every way a worker has ever died, in one place. Called both while waiting for
# the load and for 30 s after the load marker, because the KV-cache OOM lands
# after it.
check_worker_alive () {
  local tag="$1" t="$2"
  # mpirun keeps the process tree alive after its ranks error out, so waiting
  # for the PID to die burns the whole window on a run that is already lost --
  # job 315919 cost ~160 SU that way.
  #
  # The list is append-only and each entry is a job that paid for it:
  #   Cannot allocate memory / can't start new thread  -- host commit charge (315919)
  #   Executor worker died during initialization       -- CUDA set_device (316166)
  #   insufficient GPU memory / cudaMalloc             -- KV cache (316051, 316287)
  # A signature list enumerated from past failures only ever catches past
  # failures, which is why the third line is deliberately the vaguest one.
  if grep -qE "Cannot allocate memory|can't start new thread|Error executing _load_safetensors_file|Executor worker died during initialization|insufficient GPU memory|cudaMalloc" \
       "$OUT/$tag.log" 2>/dev/null; then
    echo "!!! $tag HIT A FAILURE at t=${t}s -- aborting instead of waiting out the window"
    grep -hoE "Cannot allocate memory|can't start new thread|Executor worker died during initialization|insufficient GPU memory|CUDA error: [^,]*" \
      "$OUT/$tag.log" | sort | uniq -c | sed 's/^/    /'
    exit 24
  fi
  kill -0 "${WPID[$tag]}" 2>/dev/null || {
    echo "!!! $tag DIED at t=${t}s"; tail -60 "$OUT/$tag.log"; exit 21; }
}

# Report the KV pool each worker actually got. This is INFORMATIONAL, not a
# gate, and the distinction was paid for: job 316364's prefill died with "No
# free block found" while holding 91,072 tokens against the 80,000 that N=80
# over four workers naively needs. So the naive arithmetic does not predict the
# failure -- the burst is not distributed evenly, and prefill blocks are not
# freed until their KV reaches the decode side. A gate built on that arithmetic
# would abort runs that would have worked and pass runs that die.
#
# Note also that every rank logs the allocation TWICE, a small provisional pool
# and then the real one. The larger is operative; taking the smaller reads
# 36,288 where the worker actually has 91,072.
check_kv_pool () {
  local tag="$1" pool naive
  pool=$(grep -h "tokens in paged KV cache" "$OUT/$tag.log" 2>/dev/null |
           grep -oE "\(([0-9]+)\)" | tr -d '()' | sort -n | tail -1)
  [ -n "$pool" ] || { echo "    $tag KV pool: not reported"; return 0; }
  # GUARANTEED_NO_EVICT reserves MAX_SEQ_LEN per admitted request, not the
  # tokens the request will actually use, so the figure to compare against is
  # concurrency-per-worker x MAX_SEQ_LEN. Before --max-seq-len was set the
  # engine reserved against the model's 262,144-token window and this estimate
  # was off by 57x.
  case "$tag" in
    p*) naive=$(( N * MAX_SEQ_LEN / PREFILL_WORKERS )) ;;
    *)  naive=$(( N * MAX_SEQ_LEN / DECODE_WORKERS )) ;;
  esac
  echo "    $tag KV pool $pool tokens (naive need ~$naive)"
  [ "$pool" -lt "$naive" ] && echo "    ^ below the naive figure; 316364 shows that is not decisive either way"
  return 0
}

await_worker () {
  local tag="$1" i j
  for i in $(seq 1 100); do
    # "Model init total" is logged BEFORE the KV cache is allocated, and that
    # allocation is where every prefill worker has actually been dying: 316051
    # and 316287 both show init=2, "insufficient GPU memory"=9, exit code 1, on
    # all four prefill workers. The run then hung in the smoke loop with nothing
    # alive to serve prefill -- which is what four earlier jobs recorded as a
    # "KV transceiver deadlock". Seeing the marker is not readiness. Surviving
    # 30 s past it is.
    # "Registered endpoint" and nothing earlier. Three markers have been tried
    # and the first two were each too early by a different failure:
    #   Model init total  -- precedes the KV cache allocation, and every prefill
    #                        worker died there (316051, 316287)
    #   + 30 s alive      -- precedes warmup, and at kv=0.50 the prefill workers
    #                        stalled in warmup and never registered, so the
    #                        frontend answered 503 "not ready" for 41 straight
    #                        polls while this loop believed they were up (316681)
    #   Registered endpoint -- what the frontend actually needs to route. In
    #                        316364, the run that served, all four prefill
    #                        workers logged it; in 316681 none did.
    # A readiness signal has to be later than every known way to die after it.
    if grep -q 'Registered endpoint' "$OUT/$tag.log" 2>/dev/null; then
      for j in $(seq 1 5); do
        sleep 6
        check_worker_alive "$tag" "$((i*6 + j*6))"
      done
      echo "    $tag registered and still alive at t=$((i*6 + 30))s"
      check_kv_pool "$tag"
      return 0
    fi
    check_worker_alive "$tag" "$((i*6))"
    sleep 6
  done
  echo "!!! $tag NOT LOADED in 10 min"; tail -60 "$OUT/$tag.log"; exit 22
}

# Placement: prefill takes the first 4 GPUs of node A, decode workers fill the
# rest of node A then node B. Serialised, because every TP rank mmaps the whole
# 221 GB checkpoint.
if [ "${#NODES[@]}" -lt 2 ]; then
  echo "!!! need 2 nodes, got ${#NODES[@]}"; exit 25
fi
NODE_P="${NODES[0]}"; NODE_D="${NODES[1]}"
# DECODE FIRST, and the order is load-bearing. The TP8 decode worker is the only
# load that has ever failed (315919, 315920, 316051 all died in d0 with "unable to
# mmap ~10 GB: Cannot allocate memory"); the four TP2 prefill workers have never
# failed once. Starting the risky one first means a load regression trips
# fail-fast after ~5 minutes instead of after the prefill side has already spent
# ten. The two sides register with the frontend independently, so nothing else
# depends on the order.
# All five workers start together. Decode-first-then-prefill existed to gate
# whether eight ranks could load concurrently at all; 316287 answered that
# (d0 at 108 s, p0-p3 at 138/6/12/6 s, no host OOM anywhere). They are on
# different nodes, so the only thing serialising them bought after that was
# 108 s of wall clock, which at 16 GPUs is ~29 SU.
echo; echo "########## what the model predicts for this run ##########"
# Printed BEFORE the workers start, so it cannot be written after seeing the
# answer. If the measured goodput and this disagree, one of them is wrong and
# the run has told us which question to ask.
srun -N1 -n1 -w "$NODE_A" --overlap apptainer exec $BIND "$SIF" \
  "$STAGE/trt-llm-rs" plan --total-gpus "$(( PREFILL_WORKERS * PREFILL_TP + DECODE_WORKERS * DECODE_TP ))" \
  2>&1 | head -18 | sed 's/^/  /' || echo "  (planner unavailable; continuing)"

echo; echo "########## engine.yaml validation ##########"
srun -N1 -n1 -w "$NODE_P" --overlap apptainer exec $BIND "$SIF" \
  bash -c 'for f in "$0"/engine-*.yaml; do python3 "$1" "$f" || exit 28; done' \
    "$OUT" "$SRC/scripts/validate_engine_yaml.py" || exit 28

echo; echo "########## NCCL preflight ##########"
nccl_preflight "$NODE_D" "$(( DECODE_WORKERS * DECODE_TP ))"
nccl_preflight "$NODE_P" "$(( PREFILL_WORKERS * PREFILL_TP * PREFILL_PP ))"

for w in $(seq 0 $((DECODE_WORKERS - 1))); do
  if [ "$DECODE_WORKERS" -gt 1 ]; then
    launch_worker decode "$NODE_D" "d$w" "$DECODE_TP" \
      "$(seq -s, $((w * DECODE_TP)) $((w * DECODE_TP + DECODE_TP - 1)))"
  else
    launch_worker decode "$NODE_D" "d$w" "$DECODE_TP"
  fi
done
for w in $(seq 0 $((PREFILL_WORKERS - 1))); do
  g=$(( PREFILL_TP * PREFILL_PP * CONTEXT_PARALLEL ))
  launch_worker prefill "$NODE_P" "p$w" "$PREFILL_TP" "$(seq -s, $((w * g)) $((w * g + g - 1)))" "$PREFILL_PP"
done
for w in $(seq 0 $((DECODE_WORKERS - 1))); do await_worker "d$w"; done
for w in $(seq 0 $((PREFILL_WORKERS - 1))); do await_worker "p$w"; done

########## readiness = a real request returning 200 ##########
# NOT /v1/models. In disagg the frontend lists the model before the P/D pair can
# serve and then returns 503 "not ready to serve requests yet" (A2, job 312838).
echo; echo "########## readiness: a real request, not the model list ##########"
# Do NOT gate on /v1/models. Job 316642 registered the model successfully -- the
# frontend logged "Retrying incomplete frontend model registration from
# discovery snapshot" and the MDC landed in discovery/v1/mdc -- and /v1/models
# still answered 200 in 0-4 ms with nothing this grep could extract, 50 times in
# a row. The loop never sent a single chat request and the run died having
# learned nothing. We already know the model name: we passed it in as
# --served-model-name. Use it, send the real request, and print what comes back.
MID="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
for i in $(seq 1 60); do
  if [ "$i" = 1 ]; then
    echo "  /v1/models says: $(curl -s --max-time 8 "http://$NODE_A:$HTTP_PORT/v1/models" | head -c 300)"
  fi
  if [ -n "$MID" ]; then
    S=$(curl -s --max-time 300 -w '\n%{http_code}' -X POST "http://$NODE_A:$HTTP_PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}")
    printf '%s' "$S" | grep -q '"content"' && { echo "  serving at t=$((i*10))s: $(printf '%s' "$S" | head -c 200)"; break; }
    echo "  t=$((i*10))s not serving yet: $(printf '%s' "$S" | tail -1) $(printf '%s' "$S" | head -c 160)"
  fi
  sleep 10
done
printf '%s' "${S:-}" | grep -q '"content"' || { echo "!!! NEVER SERVED"; tail -40 "$OUT/p0.log"; tail -40 "$OUT/d0.log"; exit 23; }

########## official AIPerf command ##########
echo; echo "########## official aiperf, N=$N ##########"
# Hang detector. Job 315810 deadlocked in AIPerf warmup -- returned stuck at
# 17/49 with in_flight=32 and errors=0 for 18 minutes -- because the KV transfer
# from prefill to decode never completed and TensorRT-LLM logs that timeout as
# "(observe-only)", i.e. it warns and waits forever. Left alone it would have
# burned the full 90 minute limit (~720 SU) and produced nothing. A benchmark
# that can deadlock silently needs a watchdog, not a longer time limit.
( sleep "${AIPERF_WATCHDOG_S:-420}"
  if kill -0 $$ 2>/dev/null; then
    echo "!!! WATCHDOG: aiperf exceeded ${AIPERF_WATCHDOG_S:-420}s; last progress:" >&2
    grep -E 'progress' "$OUT/aiperf.log" 2>/dev/null | tail -3 >&2
    pkill -P $$ -f aiperf 2>/dev/null
  fi ) &
WATCHDOG=$!

"$AIPERF" profile --model "$MID" --tokenizer "$MODEL" \
  --url "http://$NODE_A:$HTTP_PORT" --endpoint-type chat --endpoint /v1/chat/completions \
  --streaming --concurrency "$N" --workers-max "$N" \
  --warmup-duration 60 --benchmark-duration 120 --benchmark-grace-period 30 \
  --synthetic-input-tokens-mean 4000 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 200 --output-tokens-stddev 0 \
  --num-dataset-entries 12800 --random-seed 2026 \
  --cache-bust first_turn_prefix \
  --goodput "time_to_first_token:3000 inter_token_latency:20" \
  --ui simple --output-artifact-dir "$OUT/aiperf" > "$OUT/aiperf.log" 2>&1
AIPERF_RC=$?
kill "$WATCHDOG" 2>/dev/null
echo "  aiperf exit=$AIPERF_RC"
sed 's/\x1b\[[0-9;]*m//g' "$OUT/aiperf.log" \
  | grep -iE "Goodput|Inter Token|Time to First Token|Output Token Throughput|Request Throughput|Request Count" | sed 's/^/    /'

########## did the Rust policy decide, in every worker? ##########
kill "${PIDS[@]}" 2>/dev/null; sleep 12
echo; echo "########## bridge counters per worker ##########"
found=0
for f in "$OUT"/bridge-*.*; do
  [ -f "$f" ] || continue
  found=1
  echo "--- $(basename "$f")"
  python3 -c "
import json,sys
d=json.load(open('$f')); r=d.get('rust_stats',{})
print('   live',d['live'],'steps',d['live_steps'],'fallbacks',d['live_fallbacks'],
      'malformed',d['malformed'],'empty',d['empty_when_work_available'])
print('   cap',r.get('cap'),'not_binding',r.get('concurrency_not_binding'),
      'itl_p50',d.get('itl_p50'),'itl_p90',d.get('itl_p90'),'itl_p99',d.get('itl_p99'))
"
done
[ "$found" = 1 ] || { echo "!!! NO BRIDGE REPORT -- cannot claim the Rust policy ran"; exit 30; }
echo "artifacts: $OUT"
echo "=== done $(date -Is) ==="
