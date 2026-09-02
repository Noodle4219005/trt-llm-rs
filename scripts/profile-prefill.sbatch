#!/bin/bash
#SBATCH -J profile-both
#SBATCH -A acd110018
#SBATCH -p 8gpus
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 24
#SBATCH --mem=0
#SBATCH --gres=gpu:H200:2
#SBATCH -t 00:50:00
#SBATCH -o /work/u5727520/trt-llm-rs-runs/logs/%x-%j.out
#SBATCH -e /work/u5727520/trt-llm-rs-runs/logs/%x-%j.err
#
# The two numbers that decide what is left. See scripts/profile_prefill_decode.py
# for why each one matters.
#
# Two GPUs rather than sixteen: one TP2 worker is the prefill shape we run, and
# 2 SU/min instead of 16 buys the whole answer for about 100 SU.
set -uo pipefail
SRC=/home/u5727520/trt-llm-rs
SIF=/work/u5727520/dynamo-qwen-smoke/images/tensorrtllm-runtime-1.4.1.sif
MODEL=/work/u4063814/hf_cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969
OUT=/work/u5727520/trt-llm-rs-runs/out/profile-$SLURM_JOB_ID
BIND="--bind /work/u4063814,/work/u5727520,/home/u5727520"
mkdir -p "$OUT"

AVAIL_GB=$(df -BG --output=avail /work/u5727520 2>/dev/null | tail -1 | tr -dc '0-9')
echo "/work available: ${AVAIL_GB:-unknown} GiB"
if [ -n "${AVAIL_GB:-}" ] && [ "$AVAIL_GB" -lt "${MIN_FREE_GB:-3}" ]; then
  echo "!!! only ${AVAIL_GB} GiB free"; exit 29
fi

# The mmap loader. At TP2 each rank holds 110 GiB of weights, and without a
# shared read-only mapping two ranks on one node charge the page cache twice --
# measured at +220.2 GiB of commit against +0.0 for MAP_SHARED|PROT_READ.
cp "$SRC/scripts/usercustomize.py" "$SRC/scripts/trtllm_rs_mmap_weights.py" \
   "$SRC/scripts/profile_prefill_decode.py" "$OUT/"

echo "=== one TP2 worker: prefill then decode ==="
# No srun, and mpirun + trtllm-llmapi-launch rather than the LLM API's own
# spawn. Two cluster facts, each already paid for once: importing tensorrt_llm
# inside an srun task fails MPI_Init on a NULL communicator (it killed job
# 327547's validation step and this script's first attempt), and MPI_Comm_spawn
# is not supported by this cluster's Slurm/PMIx.
apptainer exec --nv $BIND \
  --env "MODEL=$MODEL" --env "PYTHONPATH=$OUT" \
  --env "TLLM_PROFILE_START_STOP=6-10" \
  --env "TLLM_TORCH_PROFILE_TRACE=$OUT/prefill-trace.json" \
  --env "TRTLLM_ENABLE_PDL=1" \
  "$SIF" mpirun --allow-run-as-root --oversubscribe -n 2 \
    trtllm-llmapi-launch python3 "$OUT/profile_prefill_decode.py" 2>&1 | tail -50

echo "=== artifacts ==="
ls -la "$OUT"/*.json 2>/dev/null | awk '{printf "  %s  %.1f MB\n", $9, $5/1048576}'
