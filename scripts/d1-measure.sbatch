#!/bin/bash
# S2 of the D1 experiment: ONE cell, ONE job. Parameterised by ARM and REP via
# --export, never looped inside, because independent cells that share a job turn
# max into sum and let one censored cell stall every cell behind it.
#
#   sbatch -J d1-measure-a1r-r1 --export=ALL,ARM=a1r,REP=1 scripts/d1-measure.sbatch
#   sbatch -J d1-measure-d1-r1  --export=ALL,ARM=d1,REP=1  scripts/d1-measure.sbatch
#
# ARM=a1r  frontend -> python3 -m dynamo.trtllm -> TensorRT-LLM        (2 procs)
# ARM=d1   frontend -> rust dynamo-worker -> HTTP/SSE -> python worker (3 procs)
#
# Everything except the worker is byte-identical between arms, including the
# aiperf line, so the delta is attributable. The aiperf line is copied verbatim
# from a1_dynamo_agg.sbatch's `legacy` arm, which is the A0-comparable one.
#
# Pre-registered in knowledge/experiments/2026-08-30-d1-rust-dynamo-worker-overhead.md:
# primary metric ITL mean, gate D1 <= 1.15 x A1r, three independent executions
# per arm, and a mechanism counter that must fire or the run is void.
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

: "${ARM:?ARM must be a1r or d1}"
: "${REP:?REP must be 1, 2 or 3}"
case "$ARM" in a1r|d1) ;; *) echo "!!! bad ARM '$ARM'"; exit 2 ;; esac

SIF=/work/u5727520/dynamo-qwen-smoke/images/tensorrtllm-runtime-1.4.1.sif
MODEL=/work/u4063814/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
BIND="--bind /work/u4063814,/work/u5727520,/home/u5727520"
AIPERF=/home/u5727520/hpcai26-qwen/client-venv/bin/aiperf
RUSTBIN=/work/u5727520/trt-llm-rs-runs/bin/dynamo-worker
SRC=/home/u5727520/trt-llm-rs
OUT=/work/u5727520/trt-llm-rs-runs/out/d1meas-$ARM-r$REP-$SLURM_JOB_ID
# Ports MUST be derived per job, not fixed. --mem=0 does NOT make a job
# node-exclusive on this cluster: the first wave (314686-314691) put four of six
# jobs on 25a-hgpn006 and two on 25a-hgpn004, and every one of them died with
# Dynamo's "Failed to start HTTP server: port 8100 already in use". --exclusive
# would fix it too but is the wrong tool -- it would bill the whole 8-GPU node.
port_busy () { (exec 3<>/dev/tcp/127.0.0.1/"$1") 2>/dev/null && { exec 3<&- 3>&-; return 0; }; return 1; }
free_port () { local p="$1"; while port_busy "$p"; do p=$((p + 1)); done; printf '%s' "$p"; }
HTTP_PORT=$(free_port $((20000 + (SLURM_JOB_ID * 2) % 20000)))
WORKER_PORT=$(free_port $((HTTP_PORT + 1)))
mkdir -p "$OUT"

# Co-tenancy is part of the apparatus, not a detail: a job sharing its node with
# three others does not see the same memory bandwidth as one sharing with none.
# Recorded per run so the arms can be checked for confounding instead of assumed
# comparable.
neighbours () { squeue -w "$(hostname)" -h -o '%i' 2>/dev/null | grep -vc "^$SLURM_JOB_ID$"; }
NEIGH_START=$(neighbours)

export DYN_FILE_KV="$OUT/discovery"; mkdir -p "$DYN_FILE_KV"
unset NATS_SERVER ETCD_ENDPOINTS ETCD_URL 2>/dev/null
export DYN_LOG=info

echo "=== ARM=$ARM REP=$REP node $(hostname) job $SLURM_JOB_ID $(date -Is) ==="
echo "    frontend port $HTTP_PORT  worker port $WORKER_PORT  neighbours at start: $NEIGH_START"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cleanup () {
  echo; echo "########## teardown ##########"
  kill ${RPID:-} ${WPID:-} ${FPID:-} 2>/dev/null; sleep 8
  pkill -f trtllm_dynamo_http_worker 2>/dev/null
  pkill -f dynamo-worker 2>/dev/null
  pkill -f "dynamo.trtllm" 2>/dev/null
  pkill -f "dynamo.frontend" 2>/dev/null
  sleep 5
  echo "  frontend port $HTTP_PORT: $(curl -s -o /dev/null -w '%{http_code}' --max-time 4 http://127.0.0.1:$HTTP_PORT/v1/models 2>/dev/null || echo closed)"
}
trap cleanup EXIT

########## frontend ##########
echo; echo "########## frontend on :$HTTP_PORT ##########"
apptainer exec --nv $BIND "$SIF" \
  python3 -m dynamo.frontend --http-port "$HTTP_PORT" \
    --discovery-backend file --event-plane zmq \
    > "$OUT/frontend.log" 2>&1 &
FPID=$!
sleep 15
kill -0 $FPID 2>/dev/null || { echo "!!! FRONTEND DIED"; tail -40 "$OUT/frontend.log"; exit 10; }

########## worker(s) ##########
if [ "$ARM" = "a1r" ]; then
  echo; echo "########## a1r: dynamo.trtllm worker ##########"
  apptainer exec --nv $BIND "$SIF" \
    mpirun --bind-to none --oversubscribe -H "$(hostname):12" -n 1 \
    python3 -m dynamo.trtllm \
      --model "$MODEL" --served-model-name Qwen/Qwen3-0.6B \
      --tensor-parallel-size 1 --free-gpu-memory-fraction 0.80 \
      --discovery-backend file --event-plane zmq \
      > "$OUT/worker.log" 2>&1 &
  WPID=$!
  WATCH_PIDS="$WPID"
else
  echo; echo "########## d1: python HTTP worker on :$WORKER_PORT ##########"
  ls -l "$RUSTBIN" || { echo "!!! rust binary missing"; exit 11; }
  apptainer exec --nv $BIND "$SIF" \
    mpirun --bind-to none --oversubscribe -H "$(hostname):12" -n 1 \
    python3 "$SRC/scripts/trtllm_dynamo_http_worker.py" \
      --model "$MODEL" --host 127.0.0.1 --port "$WORKER_PORT" \
      --free-gpu-memory-fraction 0.80 --tensor-parallel-size 1 \
      > "$OUT/pyworker.log" 2>&1 &
  WPID=$!
  HEALTHY=""
  for i in $(seq 1 60); do
    H=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$WORKER_PORT/health" 2>/dev/null)
    [ "$H" = "200" ] && { HEALTHY=1; echo "  python worker healthy at t=$((i*5))s"; break; }
    kill -0 $WPID 2>/dev/null || { echo "!!! PYTHON WORKER DIED"; tail -60 "$OUT/pyworker.log"; exit 12; }
    sleep 5
  done
  [ -n "$HEALTHY" ] || { echo "!!! PYTHON WORKER NEVER HEALTHY"; tail -60 "$OUT/pyworker.log"; exit 13; }

  echo; echo "########## d1: rust dynamo-worker ##########"
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
  kill -0 $RPID 2>/dev/null || { echo "!!! RUST WORKER DIED"; tail -60 "$OUT/rustworker.log"; exit 14; }
  WATCH_PIDS="$WPID $RPID"
fi

########## registration ##########
echo; echo "########## waiting for model registration ##########"
MID=""
for i in $(seq 1 60); do
  MID=$(curl -s --max-time 5 "http://127.0.0.1:$HTTP_PORT/v1/models" \
        | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
  [ -n "$MID" ] && { echo "  registered as '$MID' at t=$((i*5))s"; break; }
  for pid in $WATCH_PIDS; do
    kill -0 "$pid" 2>/dev/null || { echo "!!! A WORKER DIED at t=$((i*5))s"; tail -60 "$OUT"/*worker.log; exit 15; }
  done
  sleep 5
done
[ -n "$MID" ] || { echo "!!! NEVER REGISTERED"; tail -60 "$OUT"/*worker.log; tail -30 "$OUT/frontend.log"; exit 16; }

########## smoke before spending on aiperf ##########
echo; echo "########## smoke ##########"
ok=0
for attempt in 1 2 3; do
  S=$(curl -s --max-time 180 -w '\n%{http_code}' -X POST "http://127.0.0.1:$HTTP_PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}")
  printf '%s' "$S" | grep -q '"content"' && { ok=1; echo "  smoke ok (attempt $attempt)"; break; }
  echo "  attempt $attempt: $(printf '%s' "$S" | head -c 300)"
  sleep 20
done
[ "$ok" = 1 ] || { echo "!!! SMOKE FAILED"; tail -60 "$OUT"/*worker.log; exit 17; }

########## mechanism counter -- before ##########
if [ "$ARM" = "d1" ]; then
  BEFORE=$(grep -c 'POST /generate' "$OUT/pyworker.log" 2>/dev/null || echo 0)
else
  BEFORE=0
fi

########## measure ##########
echo; echo "########## aiperf ##########"
"$AIPERF" profile --model "$MID" --tokenizer "$MODEL" \
  --url "http://127.0.0.1:$HTTP_PORT" --endpoint-type chat --endpoint /v1/chat/completions \
  --streaming --concurrency 32 --workers-max 32 --request-count 320 --warmup-request-count 32 \
  --synthetic-input-tokens-mean 512 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 128 --output-tokens-stddev 0 \
  --extra-inputs min_tokens:128 --extra-inputs max_tokens:128 --extra-inputs ignore_eos:true \
  --extra-inputs temperature:0.0 --random-seed 2026 --ui simple --use-legacy-max-tokens \
  --output-artifact-dir "$OUT/aiperf" > "$OUT/aiperf.log" 2>&1
AIPERF_RC=$?
echo "  aiperf exit=$AIPERF_RC"
sed 's/\x1b\[[0-9;]*m//g' "$OUT/aiperf.log" \
  | grep -iE "Inter Token Latency|Time to First Token|Request Throughput|Output Token Throughput|Request Count" \
  | sed 's/^/    /'

########## mechanism counter -- after ##########
echo; echo "########## mechanism counter ##########"
if [ "$ARM" = "d1" ]; then
  AFTER=$(grep -c 'POST /generate' "$OUT/pyworker.log" 2>/dev/null || echo 0)
  FIRED=$((AFTER - BEFORE))
  echo "  POST /generate during aiperf: $FIRED (need >= 352)"
else
  FIRED=-1
  echo "  a1r: mechanism is upstream; counter is aiperf's own request count"
fi

########## machine-readable result row ##########
NEIGH_END=$(neighbours)
echo "  neighbours at end: $NEIGH_END"
CSV="$OUT/aiperf/profile_export_aiperf.csv"
# gsub the CR: aiperf writes CRLF on its single-value metric rows, and a bare
# \r smuggled into a CSV field makes Python's csv module split the row there.
val () { awk -F, -v k="$1" '$1==k {gsub(/\r/,"",$2); print $2; exit}' "$CSV" 2>/dev/null; }
{
  echo "arm,rep,job_id,node,itl_mean_ms,ttft_mean_ms,request_throughput_rps,output_token_throughput_tps,output_seq_len_mean,request_count,benchmark_duration_s,mechanism_fired,aiperf_rc,neighbours_start,neighbours_end"
  echo "$ARM,$REP,$SLURM_JOB_ID,$(hostname),$(val 'Inter Token Latency (ms)'),$(val 'Time to First Token (ms)'),$(val 'Request Throughput (requests/sec)'),$(val 'Output Token Throughput (tokens/sec)'),$(val 'Output Sequence Length (tokens)'),$(val 'Request Count'),$(val 'Benchmark Duration (sec)'),$FIRED,$AIPERF_RC,$NEIGH_START,$NEIGH_END"
} > "$OUT/result.csv"
echo; echo "--- result.csv ---"; cat "$OUT/result.csv"
echo "artifacts: $OUT"
echo "=== done $(date -Is) ==="
