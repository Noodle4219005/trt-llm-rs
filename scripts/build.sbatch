#!/bin/bash
# Build and test trt-llm-rs. CPU-only work: this workspace contains no CUDA.
#
# The GPU request is not optional. nano4's `p_dev` QOS carries
# MinTRES=gres/gpu=1, so a job with no GPU sits at QOSMinGRES forever. One GPU
# is the cheapest schedulable shape, not a resource this job uses.
#
# The account is set explicitly: a job submitted without -A can land on the
# wrong allocation and bill the wrong project.
#SBATCH -J trtllm-rs-build
#SBATCH -A acd110018
#SBATCH -p dev
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
export PATH="$CARGO_HOME/bin:$PATH"
export CARGO_BUILD_JOBS="$SLURM_CPUS_PER_TASK"
# Keep the several-GB target tree off the home quota.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/work/u5727520/.trtllm-rs-target}"
mkdir -p "$CARGO_TARGET_DIR"

echo "== host $(hostname)  cargo $(cargo --version)  target $CARGO_TARGET_DIR"

# --offline: dependencies were fetched on the login node by scripts/fetch-deps.sh.
# A compute node with no route to crates.io must fail loudly here rather than
# hang until the time limit.
#
# Every stage runs even if an earlier one fails: the point of the first job is to
# collect ALL the diagnostics in one allocation, not to stop at the first error
# and spend another one finding the next.
rc=0

echo "===== STAGE check ====="
cargo check --workspace --all-targets --offline 2>&1 || rc=1

echo "===== STAGE test ====="
# --no-fail-fast: one allocation must surface every failing crate, not just the first.
cargo test --workspace --offline --no-fail-fast 2>&1 || rc=1

echo "===== STAGE fmt ====="
cargo fmt --all -- --check 2>&1 || echo "(fmt differences only, not fatal)"

echo "===== STAGE clippy ====="
cargo clippy --workspace --all-targets --offline 2>&1 || echo "(clippy findings, not fatal on the first pass)"

# The release build and the simulations run whatever the tests did: when a test
# fails the simulated numbers are usually the thing that explains why, and
# spending another allocation to go and get them is waste.
if true; then
  echo "===== STAGE release ====="
  cargo build --release --offline 2>&1 || rc=1
  if [ "$rc" -eq 0 ]; then
    echo "===== capacity model ====="
    "$CARGO_TARGET_DIR/release/trt-llm-rs" -c configs/qwen3-4p1d.toml plan 2>&1 || rc=1
    echo "===== simulation: 4x TP2 prefill (the 4P1D shape) ====="
    "$CARGO_TARGET_DIR/release/trt-llm-rs" -c configs/qwen3-4p1d.toml sim 2>&1 || rc=1
    echo "===== simulation: 1x TP8 prefill (same 8 GPUs, one queue) ====="
    "$CARGO_TARGET_DIR/release/trt-llm-rs" -c configs/qwen3-4p1d.toml sim \
        --prefill-tp 8 --prefill-workers 1 2>&1 || rc=1
    echo "===== simulation: 2x TP4 prefill (the old 2P2D shape) ====="
    "$CARGO_TARGET_DIR/release/trt-llm-rs" -c configs/qwen3-4p1d.toml sim \
        --prefill-tp 4 --prefill-workers 2 2>&1 || rc=1
  fi
fi

echo "===== DONE rc=$rc ====="
exit "$rc"
