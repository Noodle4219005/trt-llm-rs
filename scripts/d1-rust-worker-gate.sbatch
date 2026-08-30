#!/bin/bash
# S1 of the D1 experiment: bring-up gate for the Rust Dynamo worker path.
# NO aiperf here. This job answers one question -- does a single real request
# traverse frontend -> Rust DynamoLlmEngine -> HTTP/SSE -> Python TensorRT-LLM
# worker and come back -- and it answers it in stages so a failure names itself.
#
# Every gate below is scar tissue from the A-series (see
# /work/u5727520/trt-llm-rs-runs/scripts/a1_dynamo_agg.sbatch):
#   * DYN_FILE_KV must be per-job. The file discovery backend defaults its root
#     to env::temp_dir()/dynamo_store_kv (lib/runtime/src/storage/kv.rs:163) and
#     /tmp is shared between concurrent jobs, so a stale registration from
#     another run is indistinguishable from ours. There is no CLI flag.
#   * NATS auto-enables when NATS_SERVER is set and --event-plane was left
#     unset; we set DYN_EVENT_PLANE=zmq AND unset the vars, because the
#     non-binding config path at distributed.rs:731-739 has no escape hatch.
#   * TRT-LLM's MpiPoolSession uses MPI_Comm_spawn and dies with MPI_ERR_SPAWN
#     without mpirun EVEN AT TP1. Our Python worker calls the same LLM() API,
#     so it needs the same mpirun wrapper.
#   * Readiness is the frontend LISTING the model, not a worker log line. A
#     worker that loads but never registers looks identical to a slow load.
#
# The Rust worker is configured entirely by env -- verified against the pinned
# source, not guessed: DYN_DISCOVERY_BACKEND (distributed.rs:702, default
# "etcd", valid kubernetes|etcd|file|mem) and DYN_EVENT_PLANE
# (distributed.rs:666, valid nats|zmq). dynamo_backend_common::run calls
# Runtime::from_settings(), which reads exactly those.
#SBATCH -J d1-rust-worker-gate
#SBATCH -A acd110018
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 12
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=0
#SBATCH -t 00:40:00
#SBATCH -o /work/u5727520/trt-llm-rs-runs/logs/d1-rust-worker-gate-%j.out
#SBATCH -e /work/u5727520/trt-llm-rs-runs/logs/d1-rust-worker-gate-%j.err

set -uo pipefail

SIF=/work/u5727520/dynamo-qwen-smoke/images/tensorrtllm-runtime-1.4.1.sif
MODEL=/work/u4063814/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
BIND="--bind /work/u4063814,/work/u5727520,/home/u5727520"
RUSTBIN=/work/u5727520/trt-llm-rs-runs/bin/dynamo-worker
SRC=/home/u5727520/trt-llm-rs
OUT=/work/u5727520/trt-llm-rs-runs/out/d1gate-$SLURM_JOB_ID
HTTP_PORT=8100          # Dynamo frontend
WORKER_PORT=8200        # our Python TensorRT-LLM HTTP worker
mkdir -p "$OUT"

export DYN_FILE_KV="$OUT/discovery"; mkdir -p "$DYN_FILE_KV"
unset NATS_SERVER ETCD_ENDPOINTS ETCD_URL 2>/dev/null
export DYN_LOG=info

echo "=== node $(hostname)  job $SLURM_JOB_ID  $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ls -l "$RUSTBIN" || { echo "!!! rust worker binary missing -- run scripts/d1-release-build.sbatch first"; exit 20; }

cleanup () {
  echo; echo "########## teardown ##########"
  kill ${RPID:-} ${FPID:-} ${WPID:-} 2>/dev/null
  sleep 8
  pkill -f trtllm_dynamo_http_worker 2>/dev/null
  pkill -f dynamo-worker 2>/dev/null
  pkill -f dynamo.frontend 2>/dev/null
  sleep 4
  echo "  frontend port $HTTP_PORT: $(curl -s -o /dev/null -w '%{http_code}' --max-time 4 http://127.0.0.1:$HTTP_PORT/v1/models 2>/dev/null || echo closed)"
  echo "  worker  port $WORKER_PORT: $(curl -s -o /dev/null -w '%{http_code}' --max-time 4 http://127.0.0.1:$WORKER_PORT/health 2>/dev/null || echo closed)"
}
trap cleanup EXIT

########## GATE 0 -- versions, costs nothing, answers the next failure ##########
echo; echo "########## GATE 0: versions ##########"
timeout 180 apptainer exec --nv $BIND "$SIF" python3 -c \
  "import tensorrt_llm,fastapi,uvicorn;print('trtllm',tensorrt_llm.__version__,'fastapi',fastapi.__version__,'uvicorn',uvicorn.__version__)" 2>&1 | tail -2

########## GATE 1 -- Python TensorRT-LLM HTTP worker ##########
echo; echo "########## GATE 1: python HTTP worker on :$WORKER_PORT ##########"
apptainer exec --nv $BIND "$SIF" \
  mpirun --bind-to none --oversubscribe -H "$(hostname):12" -n 1 \
  python3 "$SRC/scripts/trtllm_dynamo_http_worker.py" \
    --model "$MODEL" --host 127.0.0.1 --port "$WORKER_PORT" \
    > "$OUT/pyworker.log" 2>&1 &
WPID=$!
HEALTHY=""
for i in $(seq 1 60); do
  H=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$WORKER_PORT/health" 2>/dev/null)
  [ "$H" = "200" ] && { HEALTHY=1; echo "  healthy at t=$((i*5))s"; break; }
  kill -0 $WPID 2>/dev/null || { echo "!!! PYTHON WORKER DIED at t=$((i*5))s"; tail -60 "$OUT/pyworker.log"; exit 21; }
  sleep 5
done
[ -n "$HEALTHY" ] || { echo "!!! PYTHON WORKER NEVER HEALTHY (300s)"; tail -60 "$OUT/pyworker.log"; exit 22; }

########## GATE 2 -- the wire contract, against a REAL engine ##########
# Deliberately sends NO `extra` key. That is the exact shape the Rust
# TransportRequest emits when a request carries no backend-specific control,
# and the shape the worker used to reject with 400 "sampling is missing: extra"
# (fixed 2026-08-30). If this gate 400s, the fix regressed.
echo; echo "########## GATE 2: direct POST /generate, no extra key ##########"
TOKENS=$(timeout 180 apptainer exec $BIND "$SIF" python3 -c \
  "from transformers import AutoTokenizer;import json;t=AutoTokenizer.from_pretrained('$MODEL');print(json.dumps(t('Hello',add_special_tokens=False)['input_ids']))" 2>/dev/null | tail -1)
echo "  prompt_token_ids=$TOKENS"
[ -n "$TOKENS" ] || { echo "!!! could not tokenize"; exit 23; }
curl -s --max-time 180 -N -X POST "http://127.0.0.1:$WORKER_PORT/generate" \
  -H 'Content-Type: application/json' \
  -d "{\"request_id\":\"gate2\",\"prompt_token_ids\":$TOKENS,\"sampling\":{\"max_tokens\":8,\"temperature\":0.0,\"top_p\":1.0,\"top_k\":-1,\"ignore_eos\":false}}" \
  > "$OUT/gate2-sse.txt" 2>&1
echo "  --- first 12 SSE lines ---"; head -12 "$OUT/gate2-sse.txt" | sed 's/^/    /'
grep -q '"token_id"' "$OUT/gate2-sse.txt" || { echo "!!! GATE 2 FAILED: no token frame"; tail -40 "$OUT/pyworker.log"; exit 24; }
grep -q '"finish_reason"' "$OUT/gate2-sse.txt" || { echo "!!! GATE 2 FAILED: no terminal frame"; exit 25; }
echo "  gate 2 ok: token frames + one terminal frame, with no \`extra\` key on the wire"

########## GATE 3 -- Dynamo frontend ##########
echo; echo "########## GATE 3: frontend on :$HTTP_PORT ##########"
apptainer exec --nv $BIND "$SIF" \
  python3 -m dynamo.frontend --http-port "$HTTP_PORT" \
    --discovery-backend file --event-plane zmq \
    > "$OUT/frontend.log" 2>&1 &
FPID=$!
sleep 15
kill -0 $FPID 2>/dev/null || { echo "!!! FRONTEND DIED"; tail -40 "$OUT/frontend.log"; exit 26; }
echo "  frontend alive (pid $FPID), http $(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:$HTTP_PORT/v1/models)"

########## GATE 4 -- the Rust worker ##########
echo; echo "########## GATE 4: rust dynamo-worker ##########"
apptainer exec --nv $BIND \
  --env "TRTLLM_WORKER_URL=http://127.0.0.1:$WORKER_PORT" \
  --env "TRTLLM_MODEL=$MODEL" \
  --env "TRTLLM_SERVED_MODEL=Qwen/Qwen3-0.6B" \
  --env "TRTLLM_DEFAULT_MAX_TOKENS=128" \
  --env "DYN_DISCOVERY_BACKEND=file" \
  --env "DYN_EVENT_PLANE=zmq" \
  --env "DYN_FILE_KV=$DYN_FILE_KV" \
  --env "DYN_LOG=info" \
  "$SIF" "$RUSTBIN" > "$OUT/rustworker.log" 2>&1 &
RPID=$!
sleep 10
kill -0 $RPID 2>/dev/null || { echo "!!! RUST WORKER DIED"; tail -60 "$OUT/rustworker.log"; exit 27; }
echo "  rust worker alive (pid $RPID)"

########## GATE 5 -- registration ##########
echo; echo "########## GATE 5: waiting for model registration ##########"
MID=""
for i in $(seq 1 36); do
  MID=$(curl -s --max-time 5 "http://127.0.0.1:$HTTP_PORT/v1/models" \
        | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
  [ -n "$MID" ] && { echo "  registered as '$MID' at t=$((i*5))s"; break; }
  kill -0 $RPID 2>/dev/null || { echo "!!! RUST WORKER DIED at t=$((i*5))s"; tail -80 "$OUT/rustworker.log"; exit 28; }
  sleep 5
done
[ -n "$MID" ] || { echo "!!! NEVER REGISTERED (180s)"; echo "--- rust worker ---"; tail -80 "$OUT/rustworker.log"; echo "--- frontend ---"; tail -40 "$OUT/frontend.log"; exit 29; }

########## GATE 6 -- one real request all the way through ##########
echo; echo "########## GATE 6: chat completion through the whole chain ##########"
BEFORE=$(grep -c 'POST /generate' "$OUT/pyworker.log" 2>/dev/null || echo 0)
S=$(curl -s --max-time 180 -w '\n%{http_code}' -X POST "http://127.0.0.1:$HTTP_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}")
HC=$(printf '%s' "$S" | tail -1)
echo "  http=$HC body=$(printf '%s' "$S" | head -c 400)"
printf '%s' "$S" | grep -q '"content"' || { echo "!!! GATE 6 FAILED"; echo "--- rust ---"; tail -60 "$OUT/rustworker.log"; echo "--- py ---"; tail -40 "$OUT/pyworker.log"; exit 30; }

########## GATE 7 -- did the mechanism actually fire? ##########
# A green chat completion proves nothing about WHICH path served it. The Rust
# adapter is only proven if the Python HTTP worker's access log gained a
# POST /generate for this request.
echo; echo "########## GATE 7: mechanism counter ##########"
AFTER=$(grep -c 'POST /generate' "$OUT/pyworker.log" 2>/dev/null || echo 0)
echo "  POST /generate in pyworker.log: before=$BEFORE after=$AFTER"
[ "$AFTER" -gt "$BEFORE" ] || { echo "!!! MECHANISM DID NOT FIRE: the request did not traverse the Rust adapter"; tail -40 "$OUT/pyworker.log"; exit 31; }
echo "  gate 7 ok: the request traversed frontend -> rust -> http -> python -> trtllm"

echo; echo "=== ALL GATES PASSED $(date -Is) ==="
echo "artifacts: $OUT"
