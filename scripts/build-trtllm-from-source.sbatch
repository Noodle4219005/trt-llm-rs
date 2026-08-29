#!/bin/bash
#SBATCH -A acd110018
#SBATCH -J trtllm-from-source
#SBATCH -o /home/u5727520/trt-llm-rs/logs/trtllm-from-source-%j.out
#SBATCH -e /home/u5727520/trt-llm-rs/logs/trtllm-from-source-%j.err
#SBATCH -p 8gpus
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --exclude=25a-hgpn146,25a-hgpn047,25a-hgpn048,25a-hgpn062,25a-hgpn043,25a-hgpn071,25a-hgpn072,25a-hgpn144
#SBATCH -t 08:00:00
#
# Build TensorRT-LLM 1.3.0rc22 from source, so the final result cannot be attributed to a
# prebuilt container's optimisations. The container supplies only a toolchain here -- nvcc 13.2,
# cmake 4.0.3, ninja, g++ 13.3, TensorRT headers -- the way any build borrows a compiler. The
# artefact is ours.
#
# WHY `-p 8gpus` FOR A ONE-GPU JOB: partition names are CAPS, not requirements, and every
# partition lists the same physical nodes. `dev` cuts off at 4 hours, which this build may
# exceed; `8gpus` allows 2 days at identical billing for one GPU. Queue wait is longer.
#
# WHY ONLY 12 CORES: CPUs are hard-capped at 12 per GPU on this cluster
# ("CPU per node specification of 24 exceeds the limit (1 * 12)"), and SU is charged per
# GPU-hour. So buying cores means buying GPUs, and because build scaling is sublinear that is
# slightly MORE expensive, not less. One GPU is the cheapest configuration, just the slowest.
#
# COST: this is the single most expensive item in the project. 354 .cu and 411 .cpp files,
# ~284k lines, and the slow part is CUTLASS template instantiation (moe_gemm, fpA_intB_gemm).
# Estimate 4-8 h on 12 cores at single-arch = 240-480 SU. Stage 1 exists so that a broken
# toolchain costs ~10 minutes instead of that.
set -uo pipefail

SIF=/work/u5727520/dynamo-qwen-smoke/images/tensorrtllm-runtime-1.4.1.sif
SRC=/home/u5727520/trt-llm-rs/third_party/TensorRT-LLM
CACHE=/work/u5727520/trtllm-3rdparty-cache
CCACHE=/work/u5727520/.ccache-trtllm
# 28 TB node-local xfs. `sinfo` reports TMP_DISK=0 for every node -- that is what Slurm was told,
# not what is mounted. /home has 7 GB and /work 34 GB; neither holds a TRT-LLM build tree.
BUILD=/tmp/trtllm-build-$SLURM_JOB_ID
DIST=/work/u5727520/trtllm-wheel-$SLURM_JOB_ID
BIND="--bind /work/u5727520,/home/u5727520,/tmp"
mkdir -p "$CCACHE" "$DIST" "$BUILD"

echo "=== node $(hostname)  job $SLURM_JOB_ID  $(date -Is) ==="
df -hT /tmp | tail -1 | sed 's/^/  /'
[ -d "$CACHE" ] && echo "  3rdparty cache: $(du -sh "$CACHE" | cut -f1), $(ls -1 "$CACHE" | wc -l) repos" \
                || { echo "  !!! no 3rdparty cache at $CACHE -- run scripts/prefetch-3rdparty.sh on the LOGIN node first."; echo "  Compute nodes have no route to the internet, so FetchContent cannot clone from here."; exit 10; }

run() { apptainer exec --nv $BIND \
  --env CCACHE_DIR="$CCACHE" --env CCACHE_MAXSIZE=20G \
  "$SIF" bash -c "$1"; }

BUILD_ARGS="--build_type Release -a '90-real' -j 12 \
  --use_ccache --use-3rdparty-cache --nvtx \
  --build_dir '$BUILD' --dist_dir '$DIST' \
  -D TRTLLM_FETCHCONTENT_CACHE='$CACHE'"

echo; echo "########## STAGE 1: configure only ##########"
# Proves the toolchain, the CUDA arch, and every one of the 15 FetchContent deps resolve from the
# local cache with no network. A failure here costs ten minutes; the same failure discovered in
# stage 2 costs hours of GPU time.
T0=$SECONDS
run "cd '$SRC' && python3 scripts/build_wheel.py $BUILD_ARGS --configure-only" 2>&1 | tail -40
RC=${PIPESTATUS[0]}
echo "  stage 1 rc=$RC in $((SECONDS-T0))s"
[ "$RC" -eq 0 ] || { echo "!!! CONFIGURE FAILED -- not proceeding to the expensive stage."; exit 11; }

echo; echo "########## STAGE 2: full build ##########"
T1=$SECONDS
run "cd '$SRC' && python3 scripts/build_wheel.py $BUILD_ARGS" 2>&1 | tail -60
RC=${PIPESTATUS[0]}
echo "  stage 2 rc=$RC in $((SECONDS-T1))s ($(( (SECONDS-T1)/60 )) min)"

echo; echo "########## RESULT ##########"
ls -la "$DIST"/*.whl 2>/dev/null | sed 's/^/  /' || echo "  no wheel produced"
run "ccache -s 2>/dev/null | head -8" | sed 's/^/  /'
# Wall time is the number that makes the next estimate honest instead of a guess. Record it.
echo "  TOTAL $((SECONDS))s = $((SECONDS/60)) min = $(python3 -c "print(f'{$SECONDS/3600*60:.0f}')") SU at 1 GPU"
echo "  build tree left on node-local /tmp (not copied; it is 30-50 GB and reproducible)"
echo "=== done $(date -Is) ==="
