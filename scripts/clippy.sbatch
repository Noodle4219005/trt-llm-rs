#!/bin/bash
# Clippy over the workspace. Never run before this point, so treat the first
# output as a survey rather than a regression.
#SBATCH -J clippy
#SBATCH -A acd110018
#SBATCH -p dev
#SBATCH --gres=gpu:H200:1
#SBATCH -N 1
#SBATCH -c 12
#SBATCH -t 00:15:00
#SBATCH -o /work/u5727520/trt-llm-rs-runs/logs/%x-%j.out
#SBATCH -e /work/u5727520/trt-llm-rs-runs/logs/%x-%j.err
set -uo pipefail
SIF=/work/u5727520/dynamo-qwen-smoke/images/tensorrtllm-runtime-1.4.1.sif
apptainer exec --bind /work/u5727520,/home/u5727520,$HOME/.cargo,$HOME/.rustup "$SIF" \
  bash -lc "cd /home/u5727520/trt-llm-rs && \
    export CARGO_TARGET_DIR=/work/u5727520/.trtllm-rs-target-cpu \
      CARGO_BUILD_JOBS=\$(nproc) RUSTFLAGS='--cfg tokio_unstable' && \
    cargo clippy --workspace --exclude trtllm-pyhost --exclude trtllm-pyext \
      --all-targets 2>&1 | grep -E '^(warning|error)' | sort | uniq -c | sort -rn | head -40"
echo "exit=$?"
