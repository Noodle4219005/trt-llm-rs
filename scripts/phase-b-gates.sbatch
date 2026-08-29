#!/bin/bash
# Phase B model gates for crates/pyhost -- a real TensorRT-LLM PyTorch engine
# driven from Rust through the PyO3 boundary, on actual GPUs. Modelled on
# scripts/build-pyhost.sbatch (same container, same PyO3-needs-shared-libpython
# reasoning), extended to run the GPU-only #[ignore]'d tests in
# crates/pyhost/tests/model_gates.rs.
#
# Two H200s: gate 8 needs tensor_parallel_size=2. The exclude list is copied
# from /work/u5727520/trt-llm-rs-runs/scripts/a1_dynamo_agg.sbatch.
#
# CORRECTION VS. THE ORIGINAL TASK SPEC: gates 2/3 (tensor_parallel_size=1)
# also need the mpirun wrapper, not just gate 8. Reading
# third_party/TensorRT-LLM/tensorrt_llm/executor/executor.py:638-657 shows
# GenerationExecutor.create() falling through to GenerationExecutorProxy ->
# MpiPoolSession (MPI_Comm_spawn) even at model_world_size=1 on non-Windows;
# this matches this project's own prior finding that `python3 -m
# dynamo.trtllm` needs mpirun even at TP1 (wiki
# 2026-08-29-dynamo-layer-costs-7-percent-itl-and-its-gateway-fixes-the-aiperf-field.md,
# point 4). See crates/pyhost/tests/model_gates.rs module docs for the full
# reasoning.
#
# Each gate also runs as its OWN `cargo test <exact-name>` invocation (its
# own OS process) rather than one invocation covering gates 2/3/4 together,
# because the embedded CPython interpreter is process-global: a model loaded
# by one test would otherwise still be reachable, with its CUDA memory and
# any spawned MPI worker still live, when the next test's PyHost starts in
# the same process.
#SBATCH -J trtllm-pyhost-phase-b-gates
#SBATCH -A acd110018
#SBATCH -p dev
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --gres=gpu:H200:2
#SBATCH --mem=0
#SBATCH -t 00:40:00
#SBATCH --exclude=25a-hgpn146,25a-hgpn047,25a-hgpn048,25a-hgpn062,25a-hgpn043,25a-hgpn071,25a-hgpn072,25a-hgpn144
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

SIF=/work/u5727520/dynamo-qwen-smoke/images/tensorrtllm-runtime-1.4.1.sif
MODEL=/work/u4063814/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
# /work/u4063814 holds the model. Without it bound, the path does not exist inside the
# container, TensorRT-LLM's LLM() stops recognising it as a local directory, and falls back to
# treating it as a HuggingFace repo id:
#   HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'
# That error names the symptom and not the cause -- it reads like a bad model argument, not a
# missing mount. Job 312966 lost gates 2/3 and 8 to it.
BINDS="/work/u5727520,/work/u4063814,/home/u5727520,$HOME/.cargo,$HOME/.rustup"

# Separate from scripts/build.sbatch's CARGO_TARGET_DIR on purpose -- see
# build-pyhost.sbatch for why a host build and a container build must not
# share a target dir. Same dir AS build-pyhost.sbatch: this is the same
# container build, just a --release profile and a different test binary.
CONTAINER_TARGET_DIR=/work/u5727520/.trtllm-rs-target-container

echo "== host $(hostname)  date $(date -Is)"
echo "== sif $SIF"
echo "== model $MODEL"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

apptainer exec --nv \
  --bind "$BINDS" \
  --env "CARGO_TARGET_DIR=$CONTAINER_TARGET_DIR" \
  --env "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR" \
  --env "PYHOST_MODEL_PATH=$MODEL" \
  "$SIF" bash -euo pipefail -c '
    export PATH="$HOME/.cargo/bin:$PATH"
    mkdir -p "$CARGO_TARGET_DIR"
    cd "$SLURM_SUBMIT_DIR"

    rc=0

# A stage's verdict comes from cargo's own "test result:" line, not from the exit status.
#
# Two things make the exit status lie here, and job 312969 hit both: it reported FAILED with
# 8/8 gates passing. First, `rc` was a single cumulative variable -- once any stage set it, every
# later stage reported 1 regardless of its own outcome. Second, the model gates run under
# `mpirun`, which exits nonzero on teardown ("mpirun has exited due to process rank 0 with PID
# ...") even when the child it launched succeeded.
#
# A status signal that is wrong in the safe direction is not harmless. It trains whoever reads
# the log to ignore red, and the next red is a real one.
stage() {           # stage <label> <logfile>
  local label="$1" f="$2" res
  res=$(grep -a "^test result: " "$f" | tail -1)
  if printf '%s' "$res" | grep -q "0 failed" && printf '%s' "$res" | grep -qv "0 passed; 0 failed; 0 ignored"; then
    echo "===== $label: PASS  ($res) ====="; return 0
  elif printf '%s' "$res" | grep -q "0 failed"; then
    echo "===== $label: NO TESTS RAN  ($res) ====="; return 1
  else
    echo "===== $label: FAIL  (${res:-no test result line found}) ====="; return 1
  fi
}


    echo "===== STAGE 1: existing no-GPU boundary gates, --release (panic=abort) ====="
    cargo test -p trtllm-pyhost --release --offline -- --nocapture --test-threads=1 2>&1 || rc=1
    echo "===== STAGE 1 rc=$rc ====="

    echo "===== compile check: model_gates.rs test binary ====="
    cargo test -p trtllm-pyhost --test model_gates --release --offline --no-run 2>&1 || rc=1
    echo "===== compile check rc=$rc ====="

    echo "===== STAGE 2 (gate 2+3): real LLM load + greedy generate, TP1, under mpirun ====="
    echo "      (MpiPoolSession spawns even at TP1 -- see module docs in model_gates.rs)"
    mpirun --bind-to none --oversubscribe -H "$(hostname):24" -n 1 \
      cargo test -p trtllm-pyhost --test model_gates --release --offline \
        gate_2_3_llm_generates_text -- --ignored --exact --nocapture --test-threads=1 2>&1 || rc=1
    echo "===== STAGE 2 (gate 2+3) rc=$rc ====="

    echo "===== STAGE 2 (gate 4): CUDA stream/event ordering across separate crossings ====="
    cargo test -p trtllm-pyhost --test model_gates --release --offline \
      gate_4_cuda_stream_event_ordering_survives_crossings -- --ignored --exact --nocapture --test-threads=1 2>&1 || rc=1
    echo "===== STAGE 2 (gate 4) rc=$rc ====="

    echo "===== STAGE 3 (gate 8): TP2 multi-rank smoke, under mpirun ====="
    mpirun --bind-to none --oversubscribe -H "$(hostname):24" -n 1 \
      cargo test -p trtllm-pyhost --test model_gates --release --offline \
        gate_8_tp2_multi_rank_smoke -- --ignored --exact --nocapture --test-threads=1 2>&1 || rc=1
    echo "===== STAGE 3 (gate 8) rc=$rc ====="

    echo "===== DONE rc=$rc ====="
    exit "$rc"
  '
rc=$?
echo "===== JOB DONE rc=$rc ====="
exit "$rc"
