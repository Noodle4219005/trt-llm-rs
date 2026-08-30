#!/bin/bash
# Validate BOTH workspaces after the crates/wire extraction.
#
# There are two, and they are not built together on purpose:
#   root            crates/{core,wire,router,frontend,worker,sim,cli,...}
#   crates/dynamo   its own [workspace], so the pinned Dynamo tree (~800 crates)
#                   never enters the root resolution.
# The extraction of crates/wire is what lets crates/worker speak the worker wire
# protocol without depending on crates/dynamo -- Cargo parses the manifest of
# every optional path dependency, and dynamo-backend-common inherits `authors`
# from the Dynamo workspace root, which cannot resolve against ours.
#SBATCH -J full-workspace-validation
#SBATCH -A acd110018
#SBATCH -p dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --gres=gpu:H200:1
#SBATCH -t 01:00:00
#SBATCH -o /work/u5727520/trt-llm-rs-runs/logs/%x-%j.out
#SBATCH -e /work/u5727520/trt-llm-rs-runs/logs/%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

SIF=/work/u5727520/dynamo-qwen-smoke/images/tensorrtllm-runtime-1.4.1.sif
BINDS="/work/u4063814,/work/u5727520,/home/u5727520,$HOME/.cargo,$HOME/.rustup"
ART=/work/u5727520/trt-llm-rs-runs/out/wsval-$SLURM_JOB_ID
mkdir -p "$ART"

apptainer exec --nv --bind "$BINDS" \
  --env "CARGO_BUILD_JOBS=$SLURM_CPUS_PER_TASK" \
  --env "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR" \
  --env "ART=$ART" \
  --env "RUSTFLAGS=--cfg tokio_unstable" \
  --env "PCRE2_SYS_STATIC=1" \
  --env "LIBCLANG_PATH=/usr/lib/x86_64-linux-gnu" \
  --env "CARGO_NET_OFFLINE=true" \
  "$SIF" bash -uo pipefail -c '
    export PATH="$HOME/.cargo/bin:$PATH"
    cd "$SLURM_SUBMIT_DIR"
    rc=0

    # trtllm-pyhost is excluded, and not because its failure is inconvenient:
    # its tests/boundary.rs embeds CPython and initialises TensorRT-LLM, which
    # SIGABRTs without an mpirun wrapper (MpiPoolSession uses MPI_Comm_spawn even
    # at TP1 -- job 311967 paid 9 SU to learn that). It has its own harness,
    # scripts/build-pyhost.sbatch, which supplies the wrapper. Running it here
    # would only prove the wrapper is missing. ADR 0034 also retired pyhost from
    # the serving path, so it is not a gate on this work.
    echo "===== ROOT workspace (own target dir, pyhost has its own harness) ====="
    CARGO_TARGET_DIR=/work/u5727520/.trtllm-rs-target-container \
      cargo test --workspace --exclude trtllm-pyhost --all-targets --no-fail-fast 2>&1 | tee "$ART/root.log"
    [ "${PIPESTATUS[0]}" -ne 0 ] && rc=1

    echo "===== NESTED crates/dynamo workspace (separate target dir) ====="
    export PROTOC="$(CARGO_TARGET_DIR=/work/u5727520/.trtllm-rs-target-container \
      cargo run --manifest-path tools/protoc-bootstrap/Cargo.toml --locked --quiet)"
    CARGO_TARGET_DIR=/work/u5727520/.trtllm-rs-dynamo-target-container \
      cargo test --manifest-path crates/dynamo/Cargo.toml --features dynamo-v1 \
        --all-targets --no-fail-fast 2>&1 | tee "$ART/dynamo.log"
    [ "${PIPESTATUS[0]}" -ne 0 ] && rc=1

    exit "$rc"
  '
status=$?
./scripts/verify-source-tree.sh 2>&1 | tee "$ART/source-tree.log"
echo "--- test result lines ---"
grep -h '^test result:' "$ART"/*.log | sed 's/^/    /'
echo "artifacts: $ART"
exit "$status"
