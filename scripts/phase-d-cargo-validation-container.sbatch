#!/bin/bash
# Phase D Cargo validation for crates/dynamo, run inside the TensorRT-LLM
# runtime container instead of the bare host userspace.
#
# Two jobs established that the bare environment cannot build pinned Dynamo
# v1.4.1, for three independent reasons -- see
# knowledge/2026-08-30-pinned-dynamo-v1-4-1-needs-three-build-gates-the-container-already-has.md
#
#   314078 (57s)  protoc missing            -> etcd-client build.rs
#   314286 (35s)  libclang missing          -> nixl-sys build.rs via bindgen,
#                                              a NON-optional dep of dynamo-memory
#   314286 (35s)  --cfg tokio_unstable lost -> 9 x E0599 in dynamo-runtime
#
# The third one is the subtle one: third_party/dynamo/.cargo/config.toml sets
# rustflags = ["--cfg", "tokio_unstable"] and PCRE2_SYS_STATIC=1, but Cargo
# discovers .cargo/config.toml from the INVOCATION directory upward, never from
# a path dependency's own directory. Invoking from the repo root with
# --manifest-path crates/dynamo/Cargo.toml silently drops both. They are
# re-supplied below as env, not copied into a new config file, so the deviation
# stays visible at the call site.
#
# -C target-cpu=x86-64-v3 -C force-frame-pointers=yes are deliberately NOT
# re-supplied: correctness compilation does not need them. A benchmark build
# must revisit that, or it measures code upstream would not have emitted.
#
# The container has all three natively (measured, not assumed):
#   libclang  /usr/lib/x86_64-linux-gnu/libclang-18.so.1  (matches bindgen's
#             'libclang-*.so.*' glob, so LIBCLANG_PATH is the directory)
#   protoc    /usr/bin/protoc, libprotoc 3.21.12
#   git       2.43.0, so verify-source-tree.sh runs inside too
# The pinned tools/protoc-bootstrap is still used rather than the image's
# protoc, so the compiler version is ours and not the image's.
#
# CARGO_TARGET_DIR is separate from the host target dir on purpose: a host
# build and a container build of the same workspace must not share one, their
# linkage differs (scripts/build-pyhost.sbatch:31-34).
#
# Resource shape is forced, not chosen: p_dev carries MinTRES=gres/gpu=1 so a
# CPU-only job never admits, and one GPU caps CPUs at 1*12. Asking for more
# GPUs to buy parallelism costs slightly more, since SU is GPU-hour * 60 and
# the scaling is sublinear. Billing follows elapsed time, not the requested
# limit, so -t is generous: this is a cold target dir, ~800 crates.
#SBATCH -J trtllm-rs-phase-d-cargo-ctr
#SBATCH -A acd110018
#SBATCH -p dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --gres=gpu:H200:1
#SBATCH -t 02:00:00
#SBATCH -o /home/u5727520/trt-llm-rs/results/phase-d-cargo/runs/slurm-%x-%j.out
#SBATCH -e /home/u5727520/trt-llm-rs/results/phase-d-cargo/runs/slurm-%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

SIF=/work/u5727520/dynamo-qwen-smoke/images/tensorrtllm-runtime-1.4.1.sif
BINDS="/work/u5727520,/home/u5727520,$HOME/.cargo,$HOME/.rustup"
CONTAINER_TARGET_DIR=/work/u5727520/.trtllm-rs-dynamo-target-container
# One artifact directory per job id, so every run's logs are immutable and no
# re-run can overwrite the evidence a previous card was judged on.
ARTIFACT=/home/u5727520/trt-llm-rs/results/phase-d-cargo/runs/$SLURM_JOB_ID
mkdir -p "$ARTIFACT"

printf '== host %s  job %s  cpus %s  date %s\n' \
  "$(hostname)" "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK" "$(date -Is)"
printf '== sif %s\n== binds %s\n== container target dir %s\n' \
  "$SIF" "$BINDS" "$CONTAINER_TARGET_DIR"

apptainer exec --nv \
  --bind "$BINDS" \
  --env "CARGO_TARGET_DIR=$CONTAINER_TARGET_DIR" \
  --env "CARGO_BUILD_JOBS=$SLURM_CPUS_PER_TASK" \
  --env "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR" \
  --env "TRTLLM_RS_ARTIFACT_DIR=$ARTIFACT" \
  --env "RUSTFLAGS=--cfg tokio_unstable" \
  --env "PCRE2_SYS_STATIC=1" \
  --env "LIBCLANG_PATH=/usr/lib/x86_64-linux-gnu" \
  "$SIF" bash -euo pipefail -c '
    export PATH="$HOME/.cargo/bin:$PATH"
    mkdir -p "$CARGO_TARGET_DIR"
    cd "$SLURM_SUBMIT_DIR"

    echo "===== container toolchain ====="
    cargo --version
    rustc --version
    git --version
    ls -1 /usr/lib/x86_64-linux-gnu/libclang-18.so.1
    echo "RUSTFLAGS=$RUSTFLAGS"

    bash scripts/phase-d-cargo-validation.sh
  '
