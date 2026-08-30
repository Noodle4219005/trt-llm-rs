#!/bin/bash
# Step 3: does the Rust admission policy actually spend the ITL budget it was
# measured to be leaving on the table?
#
# The reference number this is chasing (crates/sched/src/decode.rs:3-7): 53
# sequences at a mean ITL of 17.23 ms leaves 14% of a 20 ms budget unspent, and
# that 14% is 14% of the score. This job asks the simulator whether the
# controller closes that gap, and at what cost to good_frac.
#
# No GPU work is done here -- the simulator is pure CPU. The allocation exists
# only because p_dev's QOS refuses a job with no GPU, and SLURM.md forbids
# running compute on the login node.
#
# Read `extrapolated_beyond_calibration` before quoting anything: the report
# sets it when the run drove decode past the only concurrency anyone has
# actually measured on this model, and the number is then an extrapolation of
# an unidentified curve rather than a prediction.
#SBATCH -J phase-3-sim-headroom
#SBATCH -A acd110018
#SBATCH -p dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --gres=gpu:H200:1
#SBATCH -t 00:30:00
#SBATCH -o /work/u5727520/trt-llm-rs-runs/logs/%x-%j.out
#SBATCH -e /work/u5727520/trt-llm-rs-runs/logs/%x-%j.err

set -euo pipefail

SIF=/work/u5727520/dynamo-qwen-smoke/images/tensorrtllm-runtime-1.4.1.sif
BIND="--bind /work/u5727520,/home/u5727520,$HOME/.cargo,$HOME/.rustup"
SRC=/home/u5727520/trt-llm-rs
OUT=/work/u5727520/trt-llm-rs-runs/out/phase-3-sim-$SLURM_JOB_ID
CONTAINER_TARGET_DIR=/work/u5727520/.trtllm-rs-target-container
mkdir -p "$OUT"

echo "=== node $(hostname) job $SLURM_JOB_ID $(date -Is) ==="

apptainer exec --nv $BIND \
  --env "CARGO_TARGET_DIR=$CONTAINER_TARGET_DIR" \
  --env "CARGO_BUILD_JOBS=$SLURM_CPUS_PER_TASK" \
  --env "SRC=$SRC" --env "OUT=$OUT" \
  "$SIF" bash -euo pipefail -c '
    export PATH="$HOME/.cargo/bin:$PATH"
    cd "$SRC"
    echo "===== build ====="
    cargo build --release -p trtllm-cli --offline
    BIN="$CARGO_TARGET_DIR/release/trt-llm-rs"
    ls -l "$BIN"

    echo "===== config provenance (what the numbers below assume) ====="
    "$BIN" config | tee "$OUT/config.toml"

    echo "===== reference point, JSON ====="
    "$BIN" sim --json | tee "$OUT/sim-default.json"

    echo "===== concurrency sweep x prefill policy ====="
    "$BIN" sweep --concurrency 32,48,53,64,80,96,128 \
                 --policy moore-hodgson,fcfs,edf | tee "$OUT/sweep.txt"

    echo "===== per-point JSON for the decode diagnostics ====="
    for n in 32 48 53 64 80 96 128; do
      "$BIN" sim --concurrency "$n" --json > "$OUT/sim-c$n.json"
    done
    python3 "$SRC/scripts/d1_sim_diagnostics.py" "$OUT"/sim-c*.json | tee "$OUT/decode-diagnostics.txt"
  '

echo "artifacts: $OUT"
echo "=== done $(date -Is) ==="
