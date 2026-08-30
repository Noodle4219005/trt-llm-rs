#!/bin/bash
# Stage C: Qwen3-235B-A22B-FP8 through the complete Rust control plane, with the
# official APAC HPC-AI 2026 AIPerf command.
#
#   aiperf -> trt-llm-rs serve (Rust frontend + Rust router)
#          -> HTTP/SSE -> N x trtllm_dynamo_http_worker.py (TP4 + EP4 each)
#                            PyExecutor loop untouched,
#                            self.scheduler = RustScheduler (LIVE)
#
# Parallelism is NVIDIA's own H200 recipe, not a guess:
#   recipes/qwen3-235b-a22b-fp8/trtllm/agg/hopper/deploy.yaml
#     tensor_parallel_size 4, moe_expert_parallel_size 4,
#     max-batch-size 128, max-num-tokens 8192, max-seq-len 8192,
#     kv free_gpu_memory_fraction 0.8, 4 replicas x 4 GPUs = 16
#
# WORKERS is a parameter so the same script runs the 8-GPU rehearsal (2 workers)
# and the 16-GPU scored shape (4 workers) with nothing else changed.
#
#   sbatch --export=ALL,WORKERS=2 --gres=gpu:H200:8  -N 1 scripts/stage-c-235b.sbatch
#
# N (client concurrency) is a knob, not a given: the task rules state each team
# DECLARES its own N. The score is good-request token throughput, so N has an
# optimum -- too high and good_frac collapses, too low and throughput does.
# Measured model (jobs 315449/315480/315514): a prefill-carrying iteration costs
# ~93 ms regardless of chunk size, a decode iteration ~16.4 ms, and the fraction
# of iterations carrying prefill is (concurrency per worker) / OSL. So halving N
# and doubling the replica count are the same lever.
#
# MAX_NUM_TOKENS defaults to the recipe's 8192, and the two attempts to beat
# that value are recorded because both made things worse: 1024 gave ITL 38.86 ms
# and 256 gave 86.84, against 24.37 at 8192. Prefill iteration cost does not
# scale with the tokens in the chunk -- an MoE at EP4 touches every expert and
# completes an all-to-all either way -- so a smaller chunk only multiplies the
# number of ~93 ms stalls.
#   sbatch --export=ALL,WORKERS=4 --gres=gpu:H200:8  -N 2 scripts/stage-c-235b.sbatch
#
# Timing budget: 221 GB of weights off GPFS reached serving in 8-12 minutes on
# this cluster (a5b_235b_kill_gate.sbatch's own note), so readiness is given 20
# minutes before it is called a failure.
#
# Official command, reproduced exactly, including the two things our exploratory
# runs had wrong: --random-seed 2026 (not 90001) and --cache-bust
# first_turn_prefix, which is what makes prefix reuse worthless and the router
# degenerate to least-predicted-wait.
#SBATCH -J stage-c-235b
#SBATCH -A acd110018
#SBATCH -p dev
#SBATCH --ntasks-per-node=1
#SBATCH -c 96
#SBATCH --mem=0
#SBATCH -t 01:00:00
#SBATCH -o /work/u5727520/trt-llm-rs-runs/logs/%x-%j.out
#SBATCH -e /work/u5727520/trt-llm-rs-runs/logs/%x-%j.err

set -uo pipefail
: "${WORKERS:=2}"

SIF=/work/u5727520/dynamo-qwen-smoke/images/tensorrtllm-runtime-1.4.1.sif
MODEL=/work/u4063814/hf_cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969
BIND="--bind /work/u4063814,/work/u5727520,/home/u5727520,$HOME/.cargo,$HOME/.rustup"
AIPERF=/home/u5727520/hpcai26-qwen/client-venv/bin/aiperf
SRC=/home/u5727520/trt-llm-rs
OUT=/work/u5727520/trt-llm-rs-runs/out/stage-c-$SLURM_JOB_ID
TARGET=/work/u5727520/.trtllm-rs-target-container
STAGE="$OUT/pymod"
TP=4
mkdir -p "$OUT" "$STAGE"

port_busy () { (exec 3<>/dev/tcp/127.0.0.1/"$1") 2>/dev/null && { exec 3<&- 3>&-; return 0; }; return 1; }
free_port () { local p="$1"; while port_busy "$p"; do p=$((p + 1)); done; printf '%s' "$p"; }
FRONT_PORT=$(free_port $((20000 + (SLURM_JOB_ID * 5) % 20000)))

echo "=== node $(hostname)  job $SLURM_JOB_ID  workers=$WORKERS TP=$TP EP=$TP  $(date -Is) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  /'

WPIDS=()
cleanup () {
  echo; echo "########## teardown ##########"
  kill ${RPID:-} "${WPIDS[@]}" 2>/dev/null; sleep 10
  pkill -f trtllm_dynamo_http_worker 2>/dev/null
  pkill -f 'trt-llm-rs serve' 2>/dev/null
  sleep 5
}
trap cleanup EXIT

########## build ##########
echo; echo "########## build ##########"
apptainer exec --nv $BIND \
  --env "CARGO_TARGET_DIR=$TARGET" --env "CARGO_BUILD_JOBS=16" \
  --env "PYO3_BUILD_EXTENSION_MODULE=1" --env "SRC=$SRC" --env "STAGE=$STAGE" \
  --env "CARGO_NET_OFFLINE=true" \
  "$SIF" bash -euo pipefail -c '
    export PATH="$HOME/.cargo/bin:$PATH"; cd "$SRC"
    cargo build --release -p trtllm-pyext
    cp "$CARGO_TARGET_DIR/release/libtrtllm_pyext.so" "$STAGE/trtllm_rs.so"' || exit 20
apptainer exec --nv $BIND \
  --env "CARGO_TARGET_DIR=$TARGET" --env "CARGO_BUILD_JOBS=16" \
  --env "SRC=$SRC" --env "CARGO_NET_OFFLINE=true" \
  "$SIF" bash -euo pipefail -c '
    export PATH="$HOME/.cargo/bin:$PATH"; cd "$SRC"
    cargo build --release -p trtllm-cli' || exit 21
cp "$SRC/scripts/usercustomize.py" "$SRC/scripts/trtllm_rs_shadow.py" "$STAGE/"

########## engine workers ##########
# Started ONE AT A TIME, waiting for each to become healthy before the next.
#
# Job 314880 started both at once and both died: every TP rank mmaps the whole
# 221 GB checkpoint, so 2 workers x 4 ranks hit the cgroup at the same instant
# with "unable to mmap 9998807344 bytes: Cannot allocate memory (12)" and
# "can't start new thread". Raising the limit is not available -- nano4 caps
# memory at 200 GB per GPU (`sbatch --test-only` reports
# "exceeds the limit (8 * 200 GB)"), which is 1600 GB for 8 GPUs, the same shape
# as the 12-CPU-per-GPU cap.
#
# Serialising costs much less than it looks: worker 0 pulls the 221 GB off GPFS
# into page cache, and every worker after it reads from cache rather than disk.
echo; echo "########## $WORKERS engine workers (staggered), TP$TP + EP$TP, RustScheduler LIVE ##########"
ENDPOINTS=()
for w in $(seq 0 $((WORKERS - 1))); do
  port=$(free_port $((FRONT_PORT + 100 + w)))
  gpus=$(seq -s, $((w * TP)) $((w * TP + TP - 1)))
  ENDPOINTS+=("http://127.0.0.1:$port")
  echo "  worker $w  port $port  CUDA_VISIBLE_DEVICES=$gpus"
  apptainer exec --nv $BIND \
    --env "CUDA_VISIBLE_DEVICES=$gpus" \
    --env "PYTHONPATH=$STAGE" \
    --env "TRTLLM_RS_SHADOW=1" --env "TRTLLM_RS_LIVE=1" \
    --env "TRTLLM_RS_SHADOW_OUT=$OUT/bridge-w$w" \
    --env "TRTLLM_RS_ITL_BUDGET_MS=20.0" \
    "$SIF" \
    mpirun --bind-to none --oversubscribe -H "$(hostname):48" -n 1 \
    python3 "$SRC/scripts/trtllm_dynamo_http_worker.py" \
      --model "$MODEL" --host 127.0.0.1 --port "$port" \
      --free-gpu-memory-fraction 0.80 \
      --tensor-parallel-size "$TP" --moe-expert-parallel-size "$TP" \
      --max-batch-size 128 --max-num-tokens "${MAX_NUM_TOKENS:-8192}" --max-seq-len 8192 \
      --enable-chunked-prefill \
      > "$OUT/engine-w$w.log" 2>&1 &
  WPIDS+=($!)

  # 221 GB off GPFS: 8-12 minutes cold, much less once it is in page cache.
  ok=""
  for i in $(seq 1 200); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 "${ENDPOINTS[$w]}/health" 2>/dev/null)" = "200" ] \
      && { ok=1; echo "  worker $w healthy at t=$((i*6))s"; break; }
    kill -0 "${WPIDS[$w]}" 2>/dev/null || { echo "!!! WORKER $w DIED at t=$((i*6))s"; tail -60 "$OUT/engine-w$w.log"; exit 22; }
    [ $((i % 20)) -eq 0 ] && echo "    ... worker $w t=$((i*6))s  free=$(free -g | awk '/^Mem:/{print $7}')Gi"
    sleep 6
  done
  [ -n "$ok" ] || { echo "!!! WORKER $w NOT READY in 20 min"; tail -60 "$OUT/engine-w$w.log"; exit 23; }
done
grep -h -m2 'trtllm-rs-shadow\|engine config' "$OUT"/engine-w*.log | sed 's/^/    /'

########## Rust control plane ##########
echo; echo "########## trt-llm-rs serve on :$FRONT_PORT ##########"
WORKER_ARGS=()
for e in "${ENDPOINTS[@]}"; do WORKER_ARGS+=(--worker "$e"); done
apptainer exec --nv $BIND --env "RUST_LOG=info" "$SIF" \
  "$TARGET/release/trt-llm-rs" serve \
    --addr "127.0.0.1:$FRONT_PORT" "${WORKER_ARGS[@]}" \
    --tokenizer "$MODEL/tokenizer.json" \
    --served-model-name Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
    > "$OUT/frontend.log" 2>&1 &
RPID=$!
sleep 12
kill -0 $RPID 2>/dev/null || { echo "!!! FRONTEND DIED"; tail -40 "$OUT/frontend.log"; exit 24; }
MID=$(curl -s --max-time 8 "http://127.0.0.1:$FRONT_PORT/v1/models" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
echo "  model '$MID'"
[ -n "$MID" ] || { echo "!!! NO MODEL"; tail -40 "$OUT/frontend.log"; exit 25; }

########## smoke before spending the benchmark window ##########
echo; echo "########## smoke ##########"
S=$(curl -s --max-time 300 -w '\n%{http_code}' -X POST "http://127.0.0.1:$FRONT_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}")
echo "  http=$(printf '%s' "$S" | tail -1)  $(printf '%s' "$S" | head -c 260)"
printf '%s' "$S" | grep -q '"content"' || { echo "!!! SMOKE FAILED"; tail -40 "$OUT/frontend.log"; tail -30 "$OUT"/engine-w0.log; exit 26; }

########## OFFICIAL AIPerf command ##########
echo; echo "########## official aiperf (N=${N:-32}) ##########"
"$AIPERF" profile --model "$MID" --tokenizer "$MODEL" \
  --url "http://127.0.0.1:$FRONT_PORT" --endpoint-type chat --endpoint /v1/chat/completions \
  --streaming --concurrency "${N:-32}" --workers-max "${N:-32}" \
  --warmup-duration 60 --benchmark-duration 120 --benchmark-grace-period 30 \
  --synthetic-input-tokens-mean 4000 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 200 --output-tokens-stddev 0 \
  --num-dataset-entries 12800 \
  --random-seed 2026 \
  --cache-bust first_turn_prefix \
  --goodput "time_to_first_token:3000 inter_token_latency:20" \
  --ui simple --output-artifact-dir "$OUT/aiperf" > "$OUT/aiperf.log" 2>&1
echo "  aiperf exit=$?"
sed 's/\x1b\[[0-9;]*m//g' "$OUT/aiperf.log" \
  | grep -iE "Goodput|Inter Token Latency|Time to First Token|Request Throughput|Output Token Throughput|Request Count|good" | sed 's/^/    /'

########## did the Rust policy decide? ##########
kill $RPID "${WPIDS[@]}" 2>/dev/null; sleep 12
echo; echo "########## control-plane observed ITL (one hop from the client) ##########"
grep -h 'control-plane observed' "$OUT/frontend.log" 2>/dev/null | tail -6 | sed 's/^/    /'

echo; echo "########## bridge counters ##########"
found=0
for f in "$OUT"/bridge-w*.*; do [ -f "$f" ] || continue; found=1; echo "--- $f"; cat "$f"; echo; done
[ "$found" = 1 ] || { echo "!!! NO BRIDGE REPORT -- cannot claim the Rust policy ran"; exit 30; }
echo "artifacts: $OUT"
echo "=== done $(date -Is) ==="
