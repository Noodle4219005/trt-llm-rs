#!/bin/bash
# Build and test crates/pyhost -- the PyO3 embedded-CPython boundary spike --
# inside the tensorrtllm-runtime container. Unlike scripts/build.sbatch, this
# is NOT a pure-Rust workspace build: PyO3's build.rs must resolve a Python
# 3.12 whose libpython is shared (Py_ENABLE_SHARED=1), and the only place
# that is true on this cluster is inside the container image below. The GPU
# request follows the same reasoning as build.sbatch: nano4's `p_dev` QOS
# carries MinTRES=gres/gpu=1, so one GPU is the cheapest schedulable shape,
# not a resource this build uses.
#
# The account is set explicitly: a job submitted without -A can land on the
# wrong allocation and bill the wrong project.
#SBATCH -J trtllm-pyhost-build
#SBATCH -A acd110018
#SBATCH -p dev
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --gres=gpu:H200:1
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

SIF=/work/u5727520/dynamo-qwen-smoke/images/tensorrtllm-runtime-1.4.1.sif
BINDS="/work/u5727520,/home/u5727520,$HOME/.cargo,$HOME/.rustup"

# Separate from scripts/build.sbatch's CARGO_TARGET_DIR on purpose: a host
# build and a container build of the same workspace must not share a target
# dir -- their linkage (libpython, glibc inside vs. outside the container)
# differs.
CONTAINER_TARGET_DIR=/work/u5727520/.trtllm-rs-target-container

echo "== host $(hostname)  date $(date -Is)"
echo "== sif $SIF"
echo "== binds $BINDS"
echo "== container target dir $CONTAINER_TARGET_DIR"

# Everything the inner script needs is passed in via --env rather than spliced
# into the quoted -c string, so the single-quoted body below needs no nested
# quoting at all.
apptainer exec --nv \
  --bind "$BINDS" \
  --env "CARGO_TARGET_DIR=$CONTAINER_TARGET_DIR" \
  --env "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR" \
  "$SIF" bash -euo pipefail -c '
    export PATH="$HOME/.cargo/bin:$PATH"
    mkdir -p "$CARGO_TARGET_DIR"
    cd "$SLURM_SUBMIT_DIR"

    echo "===== container python/torch/tensorrt_llm ====="
    python3 --version
    python3 -c "import sysconfig; print(\"Py_ENABLE_SHARED:\", sysconfig.get_config_var(\"Py_ENABLE_SHARED\")); print(\"LIBDIR:\", sysconfig.get_config_var(\"LIBDIR\")); print(\"INSTSONAME:\", sysconfig.get_config_var(\"INSTSONAME\"))"
    python3 -c "import torch; print(\"torch:\", torch.__version__)"
    python3 -c "import tensorrt_llm; print(\"tensorrt_llm:\", tensorrt_llm.__version__)"

    echo "===== cargo/rustc ====="
    cargo --version
    rustc --version

    echo "===== resolved PyO3 build config (for diagnosing a failure from the log alone) ====="
    # --offline: deps were resolved and vendored on the login node (cargo fetch
    # after adding this crate); a compute/container network with no route to
    # crates.io must fail loudly here rather than hang until the time limit.
    cargo build -p trtllm-pyhost --offline -vv 2>&1 | grep -i pyo3 || echo "(no pyo3-tagged build.rs output; see the STAGE build log below)"

    rc=0

    echo "===== STAGE build ====="
    cargo build -p trtllm-pyhost --offline 2>&1 || rc=1

    if [ "$rc" -eq 0 ]; then
      echo "===== STAGE test (--nocapture: prints the crossing-overhead number; --test-threads=1: one Python per process) ====="
      cargo test -p trtllm-pyhost --offline -- --nocapture --test-threads=1 2>&1 || rc=1
    fi

    echo "===== DONE rc=$rc ====="
    exit "$rc"
  '
rc=$?
echo "===== JOB DONE rc=$rc ====="
exit "$rc"
