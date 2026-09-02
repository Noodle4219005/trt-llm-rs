#!/bin/bash
#SBATCH -J vllm-profile
#SBATCH -A acd110018
#SBATCH -p 8gpus
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 72
#SBATCH --mem=0
#SBATCH --gres=gpu:H200:6
#SBATCH -t 00:40:00
#SBATCH -o /work/u5727520/trt-llm-rs-runs/logs/%x-%j.out
#SBATCH -e /work/u5727520/trt-llm-rs-runs/logs/%x-%j.err
#
# vLLM's own prefill and decode rates, measured the same way TensorRT-LLM's
# were (jobs 328143, 328182), so the comparison is like for like.
#
# The 12,050 prefill tok/s/GPU this project compares itself against is inferred
# from a teammate's 22.419 goodput, not measured. The decision to go all-in on
# vLLM rests on it, so it is worth 25 SU to check directly -- and if it does not
# reproduce, the decision needs revisiting before any code is written.
#
# No container: vLLM lives in ~/hpcai26-qwen/vllm-pixi (0.27.1, torch 2.13+cu130).
# LD_LIBRARY_PATH must point at the env's own libstdc++ because the system
# /lib64/libstdc++.so.6 lacks CXXABI_1.3.15 and sqlite3 fails to import without
# it.
set -uo pipefail
PIXI=/home/u5727520/hpcai26-qwen/vllm-pixi
ENVLIB=$PIXI/.pixi/envs/default/lib
SRC=/home/u5727520/trt-llm-rs
MODEL=/work/u4063814/hf_cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969
OUT=/work/u5727520/trt-llm-rs-runs/out/vllm-$SLURM_JOB_ID
mkdir -p "$OUT"

AVAIL_GB=$(df -BG --output=avail /work/u5727520 2>/dev/null | tail -1 | tr -dc '0-9')
echo "/work available: ${AVAIL_GB:-unknown} GiB"
[ -n "${AVAIL_GB:-}" ] && [ "$AVAIL_GB" -lt 3 ] && { echo "!!! too little disk"; exit 29; }

cp "$SRC/scripts/vllm_profile.py" "$OUT/"

run () {   # name gpus tp role [env overrides via PROF_*]
  local name=$1 gpus=$2 tp=$3 role=$4
  ( cd "$PIXI" && CUDA_VISIBLE_DEVICES="$gpus" LD_LIBRARY_PATH="$ENVLIB" \
      MODEL="$MODEL" PROF_ROLE="$role" PROF_TP="$tp" \
      PROF_BUDGET="${PROF_BUDGET:-16384}" PROF_BATCH="${PROF_BATCH:-4}" \
      PROF_EP="${PROF_EP:-0}" \
      VLLM_TORCH_PROFILER_DIR="${VLLM_TORCH_PROFILER_DIR:-}" \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      pixi run python "$OUT/vllm_profile.py" > "$OUT/$name.log" 2>&1 ) &
  echo "  launched $name gpus=$gpus tp=$tp role=$role"
}

# Four arms at once. The prefill question is whether the first measurement's
# batch shape understated the engine: 16384 tokens with a batch of 4 is right at
# the cap, and a teammate's 22.419 goodput implies 8 prefill GPUs sustained more
# than the 9,124 tok/s/GPU it produced. Doubling the budget and the batch tests
# that directly; TP4 tests whether vLLM shares TensorRT-LLM's preference for
# narrow prefill workers.
# The one configuration lever that could remove the quantisation pass.
#
# oracle/fp8.py:112-120: on Hopper with a [128,128] block-FP8 checkpoint vLLM
# keeps Triton for TP and moves FlashInfer CUTLASS to the front for EP. The
# CUTLASS path quantises activations inside the kernel
# (expects_unquantized_inputs=True) instead of in a separate per_token_group
# pass, and that separate pass is 30.86% of prefill wall clock on TensorRT-LLM.
#
# Note this runs opposite to what we learned on the other stack, where EP1 beat
# EP4 by 43%. There EP only changed how the work was split; here it changes
# which kernel does it.
# Where the remaining 2.30x to 70 req/s lives. The EP arm is the best known
# configuration (10,561 tok/s/GPU, 23.5% MFU) and gets a torch trace; the other
# two test levers that have not been measured on vLLM.
echo "=== prefill levers, with a trace on the best arm ==="
VLLM_TORCH_PROFILER_DIR="$OUT/trace-ep" \
  PROF_BUDGET=32768 PROF_BATCH=8 PROF_EP=1 run pf-ep-traced  0,1     2 prefill
PROF_BUDGET=65536 PROF_BATCH=16 PROF_EP=1 run pf-ep-64k      2,3     2 prefill
PROF_BUDGET=32768 PROF_BATCH=8  PROF_EP=1 PROF_ATTN=FLASHINFER \
  run pf-ep-fi                                               4,5     2 prefill
wait

echo; echo "=== results ==="
for n in pf-ep-traced pf-ep-64k pf-ep-fi; do
  echo "--- $n ---"
  grep -oE '(PREFILL|DECODE|RESULT) \{[^}]*\}' "$OUT/$n.log" 2>/dev/null | sed 's/^/  /' \
    || tail -15 "$OUT/$n.log" 2>/dev/null | cut -c1-160 | sed 's/^/    /'
done
