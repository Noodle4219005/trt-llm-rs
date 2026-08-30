#!/bin/bash
# S0 of the D1 experiment: build the Rust Dynamo worker in RELEASE.
#
# Debug would misrepresent the thing being measured -- the whole question is
# what the Rust adapter hop costs -- so the measured binary must be optimised.
#
# The build goes to the compute node's LOCAL /tmp, not to /work or /home:
# /work has 20G free with 12G already taken by the debug container target dir,
# and /home is at 98% (2.4G free). Compute nodes carry a 28TB node-local xfs
# /tmp that `sinfo` reports as TMP_DISK=0 -- that is what Slurm was told, not
# what is mounted. Only the ~100MB binary is copied out.
#
# crates/dynamo is its own nested workspace, so the ROOT Cargo.toml's
# [profile.release] (lto=thin, codegen-units=1, panic="abort") does NOT apply
# here; this binary gets cargo's default release profile (opt-level=3,
# lto=false, codegen-units=16, panic=unwind). Declared as a limitation in the
# experiment note rather than silently assumed away: a process-boundary cost is
# not what LTO moves. panic=unwind is also what ADR 0033 wants anyway.
#
# Resource shape is forced: p_dev's MinTRES=gres/gpu=1 rejects a CPU-only job,
# and one GPU caps CPUs at 1*12. Billing follows elapsed, not the -t limit.
#SBATCH -J d1-release-build
#SBATCH -A acd110018
#SBATCH -p dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --gres=gpu:H200:1
#SBATCH -t 01:30:00
#SBATCH -o /work/u5727520/trt-llm-rs-runs/logs/d1-release-build-%j.out
#SBATCH -e /work/u5727520/trt-llm-rs-runs/logs/d1-release-build-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

SIF=/work/u5727520/dynamo-qwen-smoke/images/tensorrtllm-runtime-1.4.1.sif
BINDS="/work/u4063814,/work/u5727520,/home/u5727520,$HOME/.cargo,$HOME/.rustup"
NODE_TARGET="/tmp/$SLURM_JOB_ID/target"
OUTBIN=/work/u5727520/trt-llm-rs-runs/bin
mkdir -p "$OUTBIN"

printf '== node %s  job %s  cpus %s  date %s\n' \
  "$(hostname)" "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK" "$(date -Is)"
df -h /tmp | tail -1

apptainer exec --nv \
  --bind "$BINDS" \
  --env "CARGO_TARGET_DIR=$NODE_TARGET" \
  --env "CARGO_BUILD_JOBS=$SLURM_CPUS_PER_TASK" \
  --env "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR" \
  --env "OUTBIN=$OUTBIN" \
  --env "RUSTFLAGS=--cfg tokio_unstable" \
  --env "PCRE2_SYS_STATIC=1" \
  --env "LIBCLANG_PATH=/usr/lib/x86_64-linux-gnu" \
  --env "CARGO_NET_OFFLINE=true" \
  "$SIF" bash -euo pipefail -c '
    export PATH="$HOME/.cargo/bin:$PATH"
    mkdir -p "$CARGO_TARGET_DIR"
    cd "$SLURM_SUBMIT_DIR"

    echo "===== toolchain ====="
    cargo --version; rustc --version

    echo "===== protoc bootstrap ====="
    export PROTOC="$(cargo run --manifest-path tools/protoc-bootstrap/Cargo.toml --locked --quiet)"
    "$PROTOC" --version

    echo "===== cargo build --release --example dynamo-worker ====="
    cargo build --release \
      --manifest-path crates/dynamo/Cargo.toml \
      --features dynamo-v1 \
      --example dynamo-worker

    src="$CARGO_TARGET_DIR/release/examples/dynamo-worker"
    ls -l "$src"
    cp "$src" "$OUTBIN/dynamo-worker"
    sha256sum "$OUTBIN/dynamo-worker"
  '

./scripts/verify-source-tree.sh
echo "== binary at $OUTBIN/dynamo-worker"
ls -l "$OUTBIN/dynamo-worker"
echo "=== done $(date -Is) ==="
