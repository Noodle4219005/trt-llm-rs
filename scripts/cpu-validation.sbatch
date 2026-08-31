#!/bin/bash
# Cargo tests. Intended for a CPU partition, but none of this user's accounts
# (acd110018, gov114009, gov115003, gov115094) are associated with ngstest,
# ngs8g or any other GRES-free partition -- sbatch --test-only rejects all four
# with "Invalid account or account/partition combination". So it runs on dev
# with one GPU, which measured 14-19 s and therefore costs about 0.3 SU.
#
# Cargo does not distribute across nodes -- it forks rustc on one machine -- so
# what matters here is the 128 cores on a single node, not the node count.
#SBATCH -J cpu-validation
#SBATCH -A acd110018
#SBATCH -p dev
#SBATCH --gres=gpu:H200:1
#SBATCH -N 1
#SBATCH -c 12
#SBATCH -t 00:10:00
#SBATCH -o /work/u5727520/trt-llm-rs-runs/logs/%x-%j.out
#SBATCH -e /work/u5727520/trt-llm-rs-runs/logs/%x-%j.err
set -uo pipefail
SIF=/work/u5727520/dynamo-qwen-smoke/images/tensorrtllm-runtime-1.4.1.sif
SRC=/home/u5727520/trt-llm-rs
TARGET=/work/u5727520/.trtllm-rs-target-cpu
echo "node=$(hostname)  cores=$(nproc)  $(date -Is)"
command -v apptainer >/dev/null || { echo "!!! no apptainer here"; exit 30; }
apptainer exec --bind /work/u5727520,/home/u5727520,$HOME/.cargo,$HOME/.rustup "$SIF" \
  bash -lc "cd $SRC && export CARGO_TARGET_DIR=$TARGET CARGO_BUILD_JOBS=\$(nproc) \
    RUSTFLAGS='--cfg tokio_unstable' && \
    cargo test --workspace --exclude trtllm-pyhost --exclude trtllm-pyext \
      --all-targets --no-fail-fast 2>&1 | tail -200 && \
    echo && echo '========== plan ==========' && \
    cargo run --quiet -p trtllm-cli -- plan --total-gpus 16 2>&1 | head -40 && \
    echo && echo '========== sim (default config) ==========' && \
    cargo run --quiet -p trtllm-cli -- sim 2>&1 | tail -30"
echo "exit=$?  $(date -Is)"
