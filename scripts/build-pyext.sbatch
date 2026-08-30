#!/bin/bash
# Build and test crates/pyext -- the PyO3 extension-module boundary, the
# opposite direction from crates/pyhost. pyhost embeds CPython inside Rust
# (feature `auto-initialize`); pyext instead builds a `.so` that a running
# CPython `dlopen()`s, with no `auto-initialize` since there is no host
# interpreter for this crate to start -- TensorRT-LLM's own Python process is
# the host. Same container as build-pyhost.sbatch for the same reason:
# resolving pyo3's build.rs and running cargo against the same Python 3.12
# TensorRT-LLM's own executor uses is only reliable inside the image below.
# The GPU request follows the same reasoning as build.sbatch /
# build-pyhost.sbatch: nano4's `p_dev` QOS carries MinTRES=gres/gpu=1, so one
# GPU is the cheapest schedulable shape, not a resource this build uses --
# the smoke test below never touches CUDA, it only imports the compiled
# module from the container's Python.
#
# DEVIATION FROM THE ORIGINAL TASK SPEC: the crate's Cargo.toml does not carry
# pyo3's `extension-module` Cargo feature. Job 313115 hit exactly why: in
# pyo3 0.29.2 that feature disables libpython linking for EVERY compile
# target in the crate, not just the cdylib -- `cargo build --release`
# succeeded but `cargo test` failed with ~20 undefined-symbol linker errors
# (PyBytes_AsString, Py_InitializeEx, ...), because a plain test binary
# cannot leave those unresolved the way a cdylib loaded by a live Python can.
# pyo3's own guide (building-and-distribution.md, faq.md, vendored at
# ~/.cargo/registry/src/*/pyo3-0.29.2/guide/) says the feature is deprecated
# for exactly this reason and to use the `PYO3_BUILD_EXTENSION_MODULE`
# env var instead, set only for the release build below -- `cargo test`
# below runs without it and links normally against the container's
# libpython (Py_ENABLE_SHARED=1, as build-pyhost.sbatch already established).
#
# The account is set explicitly: a job submitted without -A can land on the
# wrong allocation and bill the wrong project.
#
# The --exclude= list and the /work/u4063814 bind are copied from
# scripts/phase-b-gates.sbatch (itself copied from
# /work/u5727520/trt-llm-rs-runs/scripts/a1_dynamo_agg.sbatch), not from
# build-pyhost.sbatch -- build-pyhost.sbatch has neither. /work/u4063814 is
# bound for parity with that script's container mounts; this build does not
# read anything under it.
#SBATCH -J trtllm-pyext-build
#SBATCH -A acd110018
#SBATCH -p dev
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --gres=gpu:H200:1
#SBATCH -t 00:30:00
#SBATCH --exclude=25a-hgpn146,25a-hgpn047,25a-hgpn048,25a-hgpn062,25a-hgpn043,25a-hgpn071,25a-hgpn072,25a-hgpn144
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

SIF=/work/u5727520/dynamo-qwen-smoke/images/tensorrtllm-runtime-1.4.1.sif
BINDS="/work/u5727520,/work/u4063814,/home/u5727520,$HOME/.cargo,$HOME/.rustup"

# Same container target dir as build-pyhost.sbatch / phase-b-gates.sbatch on
# purpose: this is the same container build, and a host build (scripts/
# build.sbatch) must never share a target dir with a container build --
# their linkage (libpython, glibc inside vs. outside the container) differs.
CONTAINER_TARGET_DIR=/work/u5727520/.trtllm-rs-target-container

echo "== host $(hostname)  date $(date -Is)"
echo "== sif $SIF"
echo "== binds $BINDS"
echo "== container target dir $CONTAINER_TARGET_DIR"

apptainer exec --nv \
  --bind "$BINDS" \
  --env "CARGO_TARGET_DIR=$CONTAINER_TARGET_DIR" \
  --env "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR" \
  "$SIF" bash -euo pipefail -c '
    export PATH="$HOME/.cargo/bin:$PATH"
    mkdir -p "$CARGO_TARGET_DIR"
    cd "$SLURM_SUBMIT_DIR"
    mkdir -p logs

    echo "===== container python/cargo/rustc ====="
    python3 --version
    cargo --version
    rustc --version

# Each stage verdict comes from cargo own "test result: " line, not from the
# exit status -- job 312969 (crates/pyhost) reported FAILED with 8/8 gates
# passing because a cumulative `rc` variable let one stage failure mask
# every later stage real outcome. Each stage below keeps its OWN rc.
#
# `cargo test -p trtllm-pyext` runs THREE test binaries (lib unit tests,
# tests/decide_tests.rs, and an empty doctest run), so it prints multiple
# "test result: " lines. Job 313126 showed why taking only the last one is
# wrong: the trailing doctest line reads "0 passed; 0 failed; 0 ignored" (we
# have no doc examples) and that alone was read as "no tests ran", even
# though the two real lines above it said "6 passed; 0 failed" and
# "7 passed; 0 failed". Sum passed/failed across every line instead.
stage_from_log() {           # stage_from_log <label> <logfile>
  local label="$1" f="$2" lines total_passed total_failed
  lines=$(grep -a "^test result: " "$f") || true
  if [ -z "$lines" ]; then
    echo "===== $label: FAIL  (no test result line found) ====="; return 1
  fi
  total_passed=$(printf "%s\n" "$lines" | grep -oE "[0-9]+ passed" | awk "{s+=\$1} END{print s+0}") || true
  total_failed=$(printf "%s\n" "$lines" | grep -oE "[0-9]+ failed" | awk "{s+=\$1} END{print s+0}") || true
  if [ "${total_failed:-0}" -ne 0 ]; then
    echo "===== $label: FAIL  (passed=$total_passed failed=$total_failed :: $lines) ====="; return 1
  elif [ "${total_passed:-0}" -eq 0 ]; then
    echo "===== $label: NO TESTS RAN  ($lines) ====="; return 1
  else
    echo "===== $label: PASS  (passed=$total_passed failed=$total_failed) ====="; return 0
  fi
}

    echo "===== STAGE build (release, PYO3_BUILD_EXTENSION_MODULE=1 -- no libpython link) ====="
    rc_build=0
    PYO3_BUILD_EXTENSION_MODULE=1 cargo build -p trtllm-pyext --release --offline 2>&1 || rc_build=1
    echo "===== STAGE build rc=$rc_build ====="

    echo "===== STAGE test (pure-Rust decision-logic tests, no Python attached, normal libpython link) ====="
    rc_test=0
    # `|| true`: a failing test run must not trip `set -e` here -- the verdict
    # comes from stage_from_log reading the log, not from this exit status.
    # PYO3_BUILD_EXTENSION_MODULE is deliberately NOT set here -- see the
    # file header for why the test binary needs a normal libpython link.
    cargo test -p trtllm-pyext --offline -- --nocapture 2>&1 | tee logs/pyext-test.log || true
    stage_from_log "STAGE test" logs/pyext-test.log || rc_test=1

    echo "===== STAGE smoke (import the built .so from the container Python) ====="
    rc_smoke=0
    if [ "$rc_build" -eq 0 ]; then
      SO_SRC="$CARGO_TARGET_DIR/release/libtrtllm_pyext.so"
      SMOKE_DIR="$SLURM_SUBMIT_DIR/logs"
      if cp "$SO_SRC" "$SMOKE_DIR/trtllm_rs.so"; then
        # Args: ids, is_context, prompt_lens, context_done_tokens,
        # tokens_generated, max_new_tokens, arrival_ms, inflight,
        # kv_free_blocks, now_ms -- decide() now returns a 2-tuple
        # (fitting_indices, paused_indices), phase-agnostic (see src/lib.rs).
        PYTHONPATH="$SMOKE_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 -c "
import trtllm_rs
s = trtllm_rs.RustScheduler(64, 8192, 20.0, 1000)
print(s.decide([1,2],[True,False],[10,10],[0,5],[0,5],[100,100],[0.0,0.0],[],500,1.0))
print(s.stats())
" 2>&1 || rc_smoke=1
      else
        echo "cp of $SO_SRC to $SMOKE_DIR/trtllm_rs.so failed"
        rc_smoke=1
      fi
    else
      echo "skipped: STAGE build failed, no .so to smoke-test"
      rc_smoke=1
    fi
    echo "===== STAGE smoke rc=$rc_smoke ====="

    echo "===== SUMMARY build=$rc_build test=$rc_test smoke=$rc_smoke ====="
    [ "$rc_build" -eq 0 ] && [ "$rc_test" -eq 0 ] && [ "$rc_smoke" -eq 0 ]
  '
rc=$?
echo "===== JOB DONE rc=$rc ====="
exit "$rc"
