#!/bin/bash
#SBATCH -J vllm-profile
#SBATCH -A acd110018
#SBATCH -p 8gpus
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 48
#SBATCH --mem=0
#SBATCH --gres=gpu:H200:4
#SBATCH -t 00:35:00
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

run () {   # name gpus tp role
  local name=$1 gpus=$2 tp=$3 role=$4
  ( cd "$PIXI" && CUDA_VISIBLE_DEVICES="$gpus" LD_LIBRARY_PATH="$ENVLIB" \
      MODEL="$MODEL" PROF_ROLE="$role" PROF_TP="$tp" \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      pixi run python "$OUT/vllm_profile.py" > "$OUT/$name.log" 2>&1 ) &
  echo "  launched $name gpus=$gpus tp=$tp role=$role"
}

echo "=== vLLM prefill TP2 and decode TP2, side by side ==="
run prefill-tp2 0,1 2 prefill
run decode-tp2  2,3 2 decode
wait

echo; echo "=== results ==="
for n in prefill-tp2 decode-tp2; do
  echo "--- $n ---"
  grep -oE '(PREFILL|DECODE|RESULT) \{[^}]*\}' "$OUT/$n.log" 2>/dev/null | sed 's/^/  /' \
    || tail -15 "$OUT/$n.log" 2>/dev/null | cut -c1-160 | sed 's/^/    /'
done
