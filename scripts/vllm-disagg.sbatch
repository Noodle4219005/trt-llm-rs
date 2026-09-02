#!/bin/bash
#SBATCH -J vllm-disagg
#SBATCH -A acd110018
#SBATCH -p 16gpus
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 96
#SBATCH --mem=0
#SBATCH --gres=gpu:H200:8
#SBATCH -t 00:40:00
#SBATCH -o /work/u5727520/trt-llm-rs-runs/logs/%x-%j.out
#SBATCH -e /work/u5727520/trt-llm-rs-runs/logs/%x-%j.err
#
# vLLM disaggregated, configured from what we measured rather than from a
# default. Every choice below has a number behind it.
#
#   prefill  6 x TP2 with EP     10,561 tok/s/GPU, against 9,293 without EP.
#                                On Hopper with a [128,128] block-FP8
#                                checkpoint, oracle/fp8.py:112-120 sends EP to
#                                FlashInfer CUTLASS, which quantises activations
#                                inside the kernel instead of in a separate pass.
#   decode   2 x TP2 without EP  ITL 17.76 ms at C=64, against 35.06 with EP.
#                                The two roles want opposite MoE kernels, which
#                                only a split deployment can give them.
#   12 prefill / 4 decode GPUs   The measured optimum is 12.4/3.6, not the 8/8
#                                a teammate ran: prefill is 77% of the work.
#
# Predicted: 32.7 req/s and goodput near 30 without speculation. The teammate's
# measured 22.419 is the number to beat.
set -uo pipefail
PIXI=/home/u5727520/hpcai26-qwen/vllm-pixi
ENVLIB=$PIXI/.pixi/envs/default/lib
MODEL=/work/u4063814/hf_cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969
MID=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
AIPERF=/home/u5727520/hpcai26-qwen/client-venv/bin/aiperf
OUT=/work/u5727520/trt-llm-rs-runs/out/vllm-disagg-$SLURM_JOB_ID
mkdir -p "$OUT"

: "${N:=96}"
: "${PREFILL_WORKERS:=6}"
: "${DECODE_WORKERS:=2}"
: "${TP:=2}"
: "${PROXY_PORT:=8000}"

AVAIL_GB=$(df -BG --output=avail /work/u5727520 2>/dev/null | tail -1 | tr -dc '0-9')
echo "/work available: ${AVAIL_GB:-unknown} GiB"
[ -n "${AVAIL_GB:-}" ] && [ "$AVAIL_GB" -lt 3 ] && { echo "!!! too little disk"; exit 29; }

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE_P="${NODES[0]}"; NODE_D="${NODES[1]:-${NODES[0]}}"
# The proxy validates its --prefill/--decode entries as IP addresses and
# rejects hostnames: "25a-hgpn008 does not appear to be an IPv4 or IPv6
# address" (disagg_proxy_demo.py:389). It does that at argument-parse time,
# after every worker has spent fourteen minutes loading the model, so job
# 328752 cost a full allocation to learn it.
IP_P=$(getent hosts "$NODE_P" | awk '{print $1; exit}')
IP_D=$(getent hosts "$NODE_D" | awk '{print $1; exit}')
echo "prefill node $NODE_P ($IP_P)   decode node $NODE_D ($IP_D)"
[ -n "$IP_P" ] && [ -n "$IP_D" ] || { echo "!!! cannot resolve node names"; exit 30; }

PIDS=()
cleanup () {
  echo; echo "########## teardown ##########"
  kill "${PIDS[@]}" 2>/dev/null; sleep 5
  srun -N2 -n2 --overlap bash -c 'pkill -f "vllm serve"; pkill -f disagg_proxy' 2>/dev/null
  sleep 3
}
trap cleanup EXIT

# One worker. kv_role decides producer/consumer; EP is on for prefill only.
serve () {   # tag node gpus role port sidechannel
  local tag=$1 node=$2 gpus=$3 role=$4 port=$5 side=$6
  local ep="" kv
  if [ "$role" = producer ]; then ep="--enable-expert-parallel"; fi
  kv="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_${role}\"}"
  echo "  starting $tag on $node gpus=$gpus role=$role port=$port"
  srun -N1 -n1 -w "$node" --overlap bash -lc "
    cd $PIXI
    CUDA_VISIBLE_DEVICES=$gpus LD_LIBRARY_PATH=$ENVLIB \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$side UCX_NET_DEVICES=all \
    VLLM_SERVER_DEV_MODE=1 \
    pixi run vllm serve $MODEL --served-model-name $MID \
      --port $port --host 0.0.0.0 \
      --tensor-parallel-size $TP $ep \
      --max-model-len 4608 --max-num-batched-tokens 32768 \
      --gpu-memory-utilization 0.90 \
      --kv-cache-dtype fp8 --no-enable-prefix-caching \
      --kv-transfer-config '$kv' \
  " > "$OUT/$tag.log" 2>&1 &
  PIDS+=($!)
}

port_base=8100; side_base=9100

echo; echo "########## proxy argument preflight ##########"
# Run the proxy's own validator against the exact lists this job will pass, on
# the login-side python, before a single GPU touches the model. It exits
# non-zero on a bad host or port and costs nothing.
PRE_P=""; PRE_D=""
for i in $(seq 0 $((PREFILL_WORKERS-1))); do PRE_P="$PRE_P $IP_P:$((port_base+i))"; done
for i in $(seq 0 $((DECODE_WORKERS-1))); do PRE_D="$PRE_D $IP_D:$((port_base+50+i))"; done
( cd "$PIXI" && LD_LIBRARY_PATH="$ENVLIB" pixi run python - "$PRE_P" "$PRE_D" <<'PYEOF'
import sys, ipaddress
ok = True
for group, arg in (("prefill", sys.argv[1]), ("decode", sys.argv[2])):
    for inst in arg.split():
        host, _, portstr = inst.rpartition(":")
        try:
            ipaddress.ip_address(host)
            assert 0 < int(portstr) < 65536
        except Exception as e:
            print(f"  !!! {group} {inst}: {e}")
            ok = False
print("  proxy arguments valid" if ok else "  proxy arguments INVALID")
raise SystemExit(0 if ok else 1)
PYEOF
) || { echo "!!! proxy would reject these arguments; not loading 220 GiB to find out"; exit 31; }

echo; echo "########## workers ##########"
port=$port_base; side=$side_base; g=0
for i in $(seq 0 $((PREFILL_WORKERS-1))); do
  serve "p$i" "$NODE_P" "$g,$((g+1))" producer $((port+i)) $((side+i))
  g=$((g+TP))
done
g=0
for i in $(seq 0 $((DECODE_WORKERS-1))); do
  serve "d$i" "$NODE_D" "$g,$((g+1))" consumer $((port+50+i)) $((side+50+i))
  g=$((g+TP))
done

echo; echo "########## proxy ##########"
# The proxy takes host:port lists, not separate host and port flags -- checked
# against disagg_proxy_demo.py:424-440 rather than assumed. A wrong flag name
# here costs a whole 16-GPU allocation.
PLIST=""; DLIST=""
for i in $(seq 0 $((PREFILL_WORKERS-1))); do PLIST="$PLIST $IP_P:$((port+i))"; done
for i in $(seq 0 $((DECODE_WORKERS-1))); do DLIST="$DLIST $IP_D:$((port+50+i))"; done
srun -N1 -n1 -w "$NODE_P" --overlap bash -lc "
  cd $PIXI && LD_LIBRARY_PATH=$ENVLIB pixi run python \
    /work/u5727520/vllm/examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py \
    --model $MID --port $PROXY_PORT \
    --prefill $PLIST --decode $DLIST \
" > "$OUT/proxy.log" 2>&1 &
PIDS+=($!)

echo; echo "########## readiness ##########"
for i in $(seq 1 90); do
  S=$(curl -s -m 5 "http://$NODE_P:$PROXY_PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}" 2>/dev/null)
  if printf '%s' "$S" | grep -q '"content"'; then
    echo "  serving at t=$((i*10))s"; break
  fi
  [ $((i % 6)) -eq 0 ] && echo "  t=$((i*10))s not serving yet: $(printf '%s' "$S" | head -c 120)"
  sleep 10
done
printf '%s' "${S:-}" | grep -q '"content"' || {
  echo "!!! NEVER SERVED"; for f in "$OUT"/p0.log "$OUT"/d0.log "$OUT"/proxy.log; do
    echo "--- $f"; tail -25 "$f" 2>/dev/null; done; exit 23; }

echo; echo "########## official aiperf, N=$N ##########"
"$AIPERF" profile --model "$MID" --tokenizer "$MODEL" \
  --url "http://$NODE_P:$PROXY_PORT" --endpoint-type chat --endpoint /v1/chat/completions \
  --streaming --concurrency "$N" --workers-max "$N" \
  --warmup-duration 60 --benchmark-duration 120 --benchmark-grace-period 30 \
  --synthetic-input-tokens-mean 4000 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 200 --output-tokens-stddev 0 \
  --num-dataset-entries 12800 --random-seed 2026 \
  --cache-bust first_turn_prefix \
  --goodput "time_to_first_token:3000 inter_token_latency:20" \
  --ui simple --output-artifact-dir "$OUT/aiperf" > "$OUT/aiperf.log" 2>&1
echo "  aiperf exit=$?"
python3 - <<'PY' "$OUT/aiperf/profile_export_aiperf.json"
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print("  no export:", e); raise SystemExit
g = lambda k, s="avg": (d.get(k) or {}).get(s)
print(f"  goodput      {g('goodput'):8.2f} req/s")
print(f"  throughput   {g('request_throughput'):8.2f} req/s   {g('output_token_throughput'):8.0f} tok/s")
print(f"  TTFT         {g('time_to_first_token'):8.0f} avg   {g('time_to_first_token','p90'):8.0f} p90   budget 3000")
print(f"  ITL          {g('inter_token_latency'):8.2f} avg   {g('inter_token_latency','p90'):8.2f} p90   budget 20")
print(f"  requests     {g('request_count'):8.0f}")
PY
rm -f "$OUT/aiperf/inputs.json" 2>/dev/null
