#!/bin/bash
# Stage B: the complete Rust control plane against a real engine, at 0.6B.
#
#   aiperf -> trt-llm-rs serve (Rust frontend + Rust router)
#          -> HTTP/SSE -> trtllm_dynamo_http_worker.py x N (real TensorRT-LLM)
#                             PyExecutor loop untouched,
#                             self.scheduler = RustScheduler (LIVE)
#
# Every Rust crate on the serving path is exercised here: frontend, router,
# wire, worker/serving, sched (through pyext), core. This is the rehearsal for
# the 235B run and costs ~1/100th of it, which is the entire point -- the
# expensive run should be the one that only has to succeed, not the one that
# discovers the wiring is wrong.
#
# TRTLLM_RS_LIVE=1: the engine executes the Rust admission decision. A malformed
# decision falls back to upstream for that step and is counted (live_fallbacks);
# it is never executed.
#SBATCH -J stage-b-full-rust-stack
#SBATCH -A acd110018
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 12
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=0
#SBATCH -t 00:50:00
#SBATCH -o /work/u5727520/trt-llm-rs-runs/logs/%x-%j.out
#SBATCH -e /work/u5727520/trt-llm-rs-runs/logs/%x-%j.err

set -uo pipefail

SIF=/work/u5727520/dynamo-qwen-smoke/images/tensorrtllm-runtime-1.4.1.sif
MODEL=/work/u4063814/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
BIND="--bind /work/u4063814,/work/u5727520,/home/u5727520,$HOME/.cargo,$HOME/.rustup"
AIPERF=/home/u5727520/hpcai26-qwen/client-venv/bin/aiperf
SRC=/home/u5727520/trt-llm-rs
OUT=/work/u5727520/trt-llm-rs-runs/out/stage-b-$SLURM_JOB_ID
TARGET=/work/u5727520/.trtllm-rs-target-container
STAGE="$OUT/pymod"
mkdir -p "$OUT" "$STAGE"

port_busy () { (exec 3<>/dev/tcp/127.0.0.1/"$1") 2>/dev/null && { exec 3<&- 3>&-; return 0; }; return 1; }
free_port () { local p="$1"; while port_busy "$p"; do p=$((p + 1)); done; printf '%s' "$p"; }
FRONT_PORT=$(free_port $((20000 + (SLURM_JOB_ID * 3) % 20000)))
WORKER_PORT=$(free_port $((FRONT_PORT + 1)))

echo "=== node $(hostname) job $SLURM_JOB_ID  frontend :$FRONT_PORT  worker :$WORKER_PORT  $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cleanup () {
  echo; echo "########## teardown ##########"
  kill ${RPID:-} ${WPID:-} 2>/dev/null; sleep 8
  pkill -f trtllm_dynamo_http_worker 2>/dev/null
  pkill -f 'trt-llm-rs serve' 2>/dev/null
  sleep 3
}
trap cleanup EXIT

########## STAGE 1 -- build both halves ##########
echo; echo "########## STAGE 1: build ##########"
apptainer exec --nv $BIND \
  --env "CARGO_TARGET_DIR=$TARGET" --env "CARGO_BUILD_JOBS=$SLURM_CPUS_PER_TASK" \
  --env "PYO3_BUILD_EXTENSION_MODULE=1" --env "SRC=$SRC" --env "STAGE=$STAGE" \
  --env "CARGO_NET_OFFLINE=true" \
  "$SIF" bash -euo pipefail -c '
    export PATH="$HOME/.cargo/bin:$PATH"; cd "$SRC"
    cargo build --release -p trtllm-pyext
    cp "$CARGO_TARGET_DIR/release/libtrtllm_pyext.so" "$STAGE/trtllm_rs.so"
  ' || { echo "!!! pyext build failed"; exit 20; }
# The CLI must NOT be built with PYO3_BUILD_EXTENSION_MODULE set: it leaves
# libpython unresolved for every target in the crate graph (job 313115).
apptainer exec --nv $BIND \
  --env "CARGO_TARGET_DIR=$TARGET" --env "CARGO_BUILD_JOBS=$SLURM_CPUS_PER_TASK" \
  --env "SRC=$SRC" --env "CARGO_NET_OFFLINE=true" \
  "$SIF" bash -euo pipefail -c '
    export PATH="$HOME/.cargo/bin:$PATH"; cd "$SRC"
    cargo build --release -p trtllm-cli
    ls -l "$CARGO_TARGET_DIR/release/trt-llm-rs"
  ' || { echo "!!! cli build failed"; exit 21; }
cp "$SRC/scripts/usercustomize.py" "$SRC/scripts/trtllm_rs_shadow.py" "$STAGE/"

########## STAGE 2 -- Python engine worker, Rust scheduler LIVE ##########
echo; echo "########## STAGE 2: engine worker on :$WORKER_PORT (RustScheduler LIVE) ##########"
apptainer exec --nv $BIND \
  --env "PYTHONPATH=$STAGE" \
  --env "TRTLLM_RS_SHADOW=1" --env "TRTLLM_RS_LIVE=1" \
  --env "TRTLLM_RS_SHADOW_OUT=$OUT/bridge" \
  --env "TRTLLM_RS_ITL_BUDGET_MS=20.0" \
  "$SIF" \
  mpirun --bind-to none --oversubscribe -H "$(hostname):12" -n 1 \
  python3 "$SRC/scripts/trtllm_dynamo_http_worker.py" \
    --model "$MODEL" --host 127.0.0.1 --port "$WORKER_PORT" \
    --free-gpu-memory-fraction 0.80 --tensor-parallel-size 1 \
    > "$OUT/engine.log" 2>&1 &
WPID=$!
ok=""
for i in $(seq 1 60); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:$WORKER_PORT/health 2>/dev/null)" = "200" ] && { ok=1; echo "  engine healthy at t=$((i*5))s"; break; }
  kill -0 $WPID 2>/dev/null || { echo "!!! ENGINE DIED"; tail -50 "$OUT/engine.log"; exit 22; }
  sleep 5
done
[ -n "$ok" ] || { echo "!!! ENGINE NEVER HEALTHY"; tail -50 "$OUT/engine.log"; exit 23; }
grep -m3 'trtllm-rs-shadow' "$OUT/engine.log" | sed 's/^/    /'

########## STAGE 3 -- the Rust control plane ##########
echo; echo "########## STAGE 3: trt-llm-rs serve on :$FRONT_PORT ##########"
apptainer exec --nv $BIND --env "RUST_LOG=info" "$SIF" \
  "$TARGET/release/trt-llm-rs" serve \
    --addr "127.0.0.1:$FRONT_PORT" \
    --worker "http://127.0.0.1:$WORKER_PORT" \
    --tokenizer "$MODEL/tokenizer.json" \
    --served-model-name Qwen/Qwen3-0.6B \
    > "$OUT/frontend.log" 2>&1 &
RPID=$!
sleep 10
kill -0 $RPID 2>/dev/null || { echo "!!! RUST FRONTEND DIED"; tail -40 "$OUT/frontend.log"; exit 24; }
MID=$(curl -s --max-time 5 "http://127.0.0.1:$FRONT_PORT/v1/models" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
echo "  frontend lists model '$MID'"
[ -n "$MID" ] || { echo "!!! NO MODEL LISTED"; tail -40 "$OUT/frontend.log"; exit 25; }

########## STAGE 4 -- one request through the whole Rust stack ##########
echo; echo "########## STAGE 4: smoke ##########"
S=$(curl -s --max-time 180 -w '\n%{http_code}' -X POST "http://127.0.0.1:$FRONT_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}")
echo "  http=$(printf '%s' "$S" | tail -1)  body=$(printf '%s' "$S" | head -c 300)"
printf '%s' "$S" | grep -q '"content"' || { echo "!!! SMOKE FAILED"; tail -40 "$OUT/frontend.log"; tail -30 "$OUT/engine.log"; exit 26; }

########## STAGE 5 -- load ##########
echo; echo "########## STAGE 5: aiperf ##########"
"$AIPERF" profile --model "$MID" --tokenizer "$MODEL" \
  --url "http://127.0.0.1:$FRONT_PORT" --endpoint-type chat --endpoint /v1/chat/completions \
  --streaming --concurrency 32 --workers-max 32 --request-count 320 --warmup-request-count 32 \
  --synthetic-input-tokens-mean 512 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 128 --output-tokens-stddev 0 \
  --extra-inputs min_tokens:128 --extra-inputs max_tokens:128 --extra-inputs ignore_eos:true \
  --extra-inputs temperature:0.0 --random-seed 2026 --ui simple --use-legacy-max-tokens \
  --output-artifact-dir "$OUT/aiperf" > "$OUT/aiperf.log" 2>&1
echo "  aiperf exit=$?"
sed 's/\x1b\[[0-9;]*m//g' "$OUT/aiperf.log" | grep -iE "Inter Token Latency|Time to First Token|Request Throughput|Output Token Throughput|Request Count" | sed 's/^/    /'

########## STAGE 6 -- did the Rust policy actually decide? ##########
kill $RPID $WPID 2>/dev/null; sleep 10
echo; echo "########## STAGE 6: bridge counters ##########"
found=0
for f in "$OUT"/bridge.*; do [ -f "$f" ] || continue; found=1; echo "--- $f"; cat "$f"; done
[ "$found" = 1 ] || { echo "!!! NO BRIDGE REPORT -- cannot claim the Rust policy ran"; exit 30; }
echo "artifacts: $OUT"
echo "=== done $(date -Is) ==="
