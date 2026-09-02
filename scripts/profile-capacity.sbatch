#!/bin/bash
#SBATCH -J profile-cap
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
# Three measurements at once. The wall clock of this job is one model load,
# because loading 220 GiB from cold GPFS dominates and all three workers pay it
# concurrently rather than in series. A 2-GPU version of the same questions was
# still loading at 30 minutes.
#
#   node A, GPUs 0-1   mixed TP2   in-card P/D: one worker doing both, which is
#   node A, GPUs 2-5   mixed TP4   the only way to reach a fractional split.
#
# The optimal prefill:decode allocation moves as prefill improves -- 10.9/5.1
# GPUs at our measured 15.1% prefill MFU, 5.7/10.3 at the 59.3% a 38 req/s
# target implies -- so a physical pool split must be re-cut whenever the kernels
# change, while a shared card only changes a token-budget ratio. It also removes
# the KV transfer: the cache is already where decode needs it.
#
# Pre-registered: per-request GPU-seconds at the measured rates are
# 4000/6804 + 200/726 = 0.863, so sixteen perfectly packed GPUs give 18.53 req/s
# and one TP2 replica of eight should give 2.32.   the number nobody has: saturated decode
#                                    capacity. Every disaggregated run was
#                                    prefill-bound so this side never reached
#                                    its ceiling, and it is exactly what decides
#                                    whether a vLLM-prefill + TRT-LLM-decode
#                                    hybrid could reach the target.
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

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE_A="${NODES[0]}"; NODE_B="$NODE_A"
echo "prefill node $NODE_A   decode node $NODE_B"

cp "$SRC/scripts/usercustomize.py" "$SRC/scripts/trtllm_rs_mmap_weights.py" \
   "$SRC/scripts/profile_prefill_decode.py" "$OUT/"

# Exactly the launch the working disaggregated script uses: mpirun -n 1 with
# -H node:96, letting TRT-LLM spawn its own TP ranks.
#
# Job 328038 used `-n $tp` with trtllm-llmapi-launch instead and hung for forty
# minutes: both processes logged "rank 0 using MpiCommSession", i.e. two
# separate MPI worlds of size one, so the TP group never formed and each rank
# waited for a peer that did not exist. An older note recommending
# trtllm-llmapi-launch came from a different container and does not apply here.
#
# TRTLLM_RS_MMAP_WEIGHTS=1 is also from that launcher and was missing:
# without it each rank reads its shard rather than sharing a MAP_SHARED
# mapping, which was measured at +220.2 GiB of commit charge against +0.0.
launch () {                       # name node gpus tp role
  local name=$1 node=$2 gpus=$3 tp=$4 role=$5
  local chunk=${PROF_CHUNK:-0}
  srun -N1 -n1 -w "$node" --overlap apptainer exec --nv $BIND \
    --env "CUDA_VISIBLE_DEVICES=$gpus" \
    --env "MODEL=$MODEL" --env "PYTHONPATH=$OUT" \
    --env "PROF_ROLE=$role" --env "PROF_TP=$tp" --env "PROF_CHUNK=$chunk" \
    --env "TLLM_PROFILE_START_STOP=6-10" \
    --env "TLLM_TORCH_PROFILE_TRACE=$OUT/$name-trace.json" \
    --env "TRTLLM_ENABLE_PDL=1" \
    --env "TRTLLM_RS_MMAP_WEIGHTS=1" \
    "$SIF" mpirun --bind-to none --oversubscribe -H "$node:48" -n 1 \
      python3 "$OUT/profile_prefill_decode.py" \
      > "$OUT/$name.log" 2>&1 &
  echo "  launched $name on $node gpus=$gpus tp=$tp role=$role"
}

echo "=== launching three workers ==="
PROF_CHUNK=1280 launch chunk1280 "$NODE_A" 0,1 2 mixed
PROF_CHUNK=2560 launch chunk2560 "$NODE_A" 2,3 2 mixed
wait

echo; echo "=== results ==="
for n in chunk1280 chunk2560; do
  echo "--- $n ---"
  grep -E "^(PREFILL|DECODE|RESULT) " "$OUT/$n.log" 2>/dev/null | sed 's/^/  /' \
    || echo "  (no result; tail follows)"
  grep -E "^(PREFILL|DECODE|RESULT) " "$OUT/$n.log" >/dev/null 2>&1 \
    || tail -12 "$OUT/$n.log" 2>/dev/null | cut -c1-160 | sed 's/^/    /'
done

echo; echo "=== artifacts ==="
ls -la "$OUT"/*.json 2>/dev/null | awk '{printf "  %s  %.1f MB\n", $9, $5/1048576}'
