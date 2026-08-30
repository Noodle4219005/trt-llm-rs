#!/bin/bash
# Phase C shadow gate: run the Rust capacity policy beside upstream's, on GPU,
# under a real 320-request aiperf workload. THE GPU ONLY EXECUTES UPSTREAM'S
# DECISION -- this job measures how far the Rust policy would have diverged, and
# whether it ever produces something malformed.
#
# ADR 0034 fixes both the hook point and how to read the result:
#   * hook point  = the capacity scheduler (_util.py:2433), not the whole
#     RequestScheduler and not the executor loop.
#   * `diverged`  = the EXPECTED SIGNAL. Upstream is GuaranteedNoEvict; ours is
#     an ITL-budget admission controller, and the project exists because they
#     should disagree. **Zero divergence is the failure to worry about**: it is
#     indistinguishable from "the shim never installed".
#   * `malformed` and `empty_when_work_available` are the real failures.
#
# The shim is installed through `usercustomize` on PYTHONPATH because the
# executor runs in an MPI_Comm_spawn'd interpreter that does not inherit the
# launching process's monkeypatch. It patches only after
# tensorrt_llm._torch.pyexecutor._util finishes importing, so no interpreter on
# this node pays for importing TensorRT-LLM it did not already want.
#
# Nothing under third_party/ is edited.
#SBATCH -J phase-c-shadow
#SBATCH -A acd110018
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 12
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=0
#SBATCH -t 00:40:00
#SBATCH -o /work/u5727520/trt-llm-rs-runs/logs/%x-%j.out
#SBATCH -e /work/u5727520/trt-llm-rs-runs/logs/%x-%j.err

set -uo pipefail

SIF=/work/u5727520/dynamo-qwen-smoke/images/tensorrtllm-runtime-1.4.1.sif
MODEL=/work/u4063814/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
BIND="--bind /work/u4063814,/work/u5727520,/home/u5727520,$HOME/.cargo,$HOME/.rustup"
AIPERF=/home/u5727520/hpcai26-qwen/client-venv/bin/aiperf
SRC=/home/u5727520/trt-llm-rs
OUT=/work/u5727520/trt-llm-rs-runs/out/phase-c-shadow-$SLURM_JOB_ID
CONTAINER_TARGET_DIR=/work/u5727520/.trtllm-rs-target-container
STAGE="$OUT/pymod"
mkdir -p "$OUT" "$STAGE"

port_busy () { (exec 3<>/dev/tcp/127.0.0.1/"$1") 2>/dev/null && { exec 3<&- 3>&-; return 0; }; return 1; }
free_port () { local p="$1"; while port_busy "$p"; do p=$((p + 1)); done; printf '%s' "$p"; }
HTTP_PORT=$(free_port $((20000 + (SLURM_JOB_ID * 2) % 20000)))

export DYN_FILE_KV="$OUT/discovery"; mkdir -p "$DYN_FILE_KV"
unset NATS_SERVER ETCD_ENDPOINTS ETCD_URL 2>/dev/null
export DYN_LOG=info

echo "=== node $(hostname) job $SLURM_JOB_ID port $HTTP_PORT $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

########## STAGE 1 -- build the extension module, fresh ##########
# Rebuilt rather than reused: an .so from an earlier day is indistinguishable
# from a current one in a log, and a stale policy would be reported as this
# commit's behaviour. PYO3_BUILD_EXTENSION_MODULE is set ONLY here -- as a
# Cargo feature it would break `cargo test` linking (job 313115).
echo; echo "########## STAGE 1: build trtllm_rs ##########"
apptainer exec --nv $BIND \
  --env "CARGO_TARGET_DIR=$CONTAINER_TARGET_DIR" \
  --env "CARGO_BUILD_JOBS=$SLURM_CPUS_PER_TASK" \
  --env "PYO3_BUILD_EXTENSION_MODULE=1" \
  --env "SRC=$SRC" --env "STAGE=$STAGE" \
  "$SIF" bash -euo pipefail -c '
    export PATH="$HOME/.cargo/bin:$PATH"
    cd "$SRC"
    cargo build --release -p trtllm-pyext --offline
    cp "$CARGO_TARGET_DIR/release/libtrtllm_pyext.so" "$STAGE/trtllm_rs.so"
    ls -l "$STAGE/trtllm_rs.so"
  ' || { echo "!!! STAGE 1 FAILED"; exit 20; }
cp "$SRC/scripts/usercustomize.py" "$SRC/scripts/trtllm_rs_shadow.py" "$STAGE/"

########## STAGE 2 -- import smoke, before spending on the engine ##########
echo; echo "########## STAGE 2: import smoke ##########"
apptainer exec --nv $BIND --env "PYTHONPATH=$STAGE" "$SIF" \
  python3 -c "import trtllm_rs; s=trtllm_rs.RustScheduler(16,8192,20.0,1024); print('RustScheduler ok, stats:', s.stats())" \
  || { echo "!!! STAGE 2 FAILED: extension not importable"; exit 21; }

########## STAGE 3 -- frontend ##########
echo; echo "########## STAGE 3: frontend on :$HTTP_PORT ##########"
apptainer exec --nv $BIND "$SIF" \
  python3 -m dynamo.frontend --http-port "$HTTP_PORT" \
    --discovery-backend file --event-plane zmq > "$OUT/frontend.log" 2>&1 &
FPID=$!
sleep 15
kill -0 $FPID 2>/dev/null || { echo "!!! FRONTEND DIED"; tail -40 "$OUT/frontend.log"; exit 22; }

########## STAGE 4 -- worker WITH the shadow armed ##########
echo; echo "########## STAGE 4: dynamo.trtllm worker, shadow armed ##########"
apptainer exec --nv $BIND \
  --env "PYTHONPATH=$STAGE" \
  --env "TRTLLM_RS_SHADOW=1" \
  --env "TRTLLM_RS_SHADOW_OUT=$OUT/shadow" \
  --env "TRTLLM_RS_ITL_BUDGET_MS=20.0" \
  "$SIF" \
  mpirun --bind-to none --oversubscribe -H "$(hostname):12" -n 1 \
  python3 -m dynamo.trtllm \
    --model "$MODEL" --served-model-name Qwen/Qwen3-0.6B \
    --tensor-parallel-size 1 --free-gpu-memory-fraction 0.80 \
    --discovery-backend file --event-plane zmq \
    > "$OUT/worker.log" 2>&1 &
WPID=$!

MID=""
for i in $(seq 1 60); do
  MID=$(curl -s --max-time 5 "http://127.0.0.1:$HTTP_PORT/v1/models" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
  [ -n "$MID" ] && { echo "  registered as '$MID' at t=$((i*5))s"; break; }
  kill -0 $WPID 2>/dev/null || { echo "!!! WORKER DIED"; tail -60 "$OUT/worker.log"; kill $FPID; exit 23; }
  sleep 5
done
[ -n "$MID" ] || { echo "!!! NEVER REGISTERED"; tail -60 "$OUT/worker.log"; kill $WPID $FPID; exit 24; }

echo "--- shadow install lines from the worker log ---"
grep -n 'trtllm-rs-shadow' "$OUT/worker.log" | head -10 | sed 's/^/    /'

########## STAGE 5 -- real workload ##########
echo; echo "########## STAGE 5: aiperf ##########"
"$AIPERF" profile --model "$MID" --tokenizer "$MODEL" \
  --url "http://127.0.0.1:$HTTP_PORT" --endpoint-type chat --endpoint /v1/chat/completions \
  --streaming --concurrency 32 --workers-max 32 --request-count 320 --warmup-request-count 32 \
  --synthetic-input-tokens-mean 512 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 128 --output-tokens-stddev 0 \
  --extra-inputs min_tokens:128 --extra-inputs max_tokens:128 --extra-inputs ignore_eos:true \
  --extra-inputs temperature:0.0 --random-seed 2026 --ui simple --use-legacy-max-tokens \
  --output-artifact-dir "$OUT/aiperf" > "$OUT/aiperf.log" 2>&1
echo "  aiperf exit=$?"
sed 's/\x1b\[[0-9;]*m//g' "$OUT/aiperf.log" | grep -iE "Inter Token Latency|Request Throughput|Request Count" | sed 's/^/    /'

########## teardown, then read the counters ##########
kill $WPID $FPID 2>/dev/null; sleep 12
pkill -f "dynamo.trtllm" 2>/dev/null; pkill -f "dynamo.frontend" 2>/dev/null; sleep 4

echo; echo "########## shadow counters ##########"
found=0
for f in "$OUT"/shadow.*; do
  [ -f "$f" ] || continue
  found=1
  echo "--- $f"; cat "$f"
done
if [ "$found" = 0 ]; then
  echo "!!! NO SHADOW REPORT WRITTEN -- the shim did not run. Any 'agreement' claim would be false."
  grep -n 'trtllm-rs-shadow' "$OUT/worker.log" | tail -20
  exit 30
fi
echo "artifacts: $OUT"
echo "=== done $(date -Is) ==="
