#!/bin/bash
#SBATCH -A acd110018
#SBATCH -J trtllm-rs-shadow
#SBATCH -o /home/u5727520/trt-llm-rs/logs/shadow-%j.out
#SBATCH -e /home/u5727520/trt-llm-rs/logs/shadow-%j.err
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --exclude=25a-hgpn146,25a-hgpn047,25a-hgpn048,25a-hgpn062,25a-hgpn043,25a-hgpn071,25a-hgpn072,25a-hgpn144
#SBATCH -t 00:30:00
#
# C0b -- the shadow gate. Run a real serving workload with the Rust scheduler deciding ALONGSIDE
# TensorRT-LLM's own, and compare. The GPU eats the upstream decision throughout, so this run's
# performance is by construction identical to a stock run; the only output is whether the two
# schedulers agree.
#
# This is the cheapest correctness gate in the project: it needs no separate model, no separate
# workload, and can ride along with any job. Getting agreement on a few hundred thousand real
# iterations before switching to `live` is worth far more than any amount of unit testing, because
# the inputs are real request mixes rather than ones we thought to imagine.
#
# TRTLLM_RS_MODE=live flips it to Rust actually driving, for the A/B that follows.
set -uo pipefail

SIF=/work/u5727520/dynamo-qwen-smoke/images/tensorrtllm-runtime-1.4.1.sif
MODEL=/work/u4063814/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
REPO=/home/u5727520/trt-llm-rs
BIND="--bind /work/u4063814,/work/u5727520,/home/u5727520"
AIPERF=/home/u5727520/hpcai26-qwen/client-venv/bin/aiperf
OUT=/work/u5727520/trt-llm-rs-runs/out/shadow-$SLURM_JOB_ID
PORT=8400
MODE=${TRTLLM_RS_MODE:-shadow}
mkdir -p "$OUT"

# The .so is built by scripts/build-pyext.sbatch into the container target dir. Stage it under a
# name Python can import, next to the pure-Python bridge.
SO=$(ls -1t /work/u5727520/.trtllm-rs-target-container/release/libtrtllm_pyext.so 2>/dev/null | head -1)
[ -n "$SO" ] || { echo "!!! no libtrtllm_pyext.so -- run scripts/build-pyext.sbatch first"; exit 10; }
mkdir -p "$OUT/pypath"
cp "$SO" "$OUT/pypath/trtllm_rs.so"
cp -r "$REPO/python/trtllm_rs_bridge" "$OUT/pypath/"
echo "=== node $(hostname)  job $SLURM_JOB_ID  mode=$MODE  $(date -Is) ==="
echo "  so:     $SO"
echo "  bridge: $REPO/python/trtllm_rs_bridge"

cat > "$OUT/engine.yaml" <<'YAML'
disable_overlap_scheduler: false
YAML

# Gate 0: the module imports and the class instantiates, before spending any model-load time.
apptainer exec --nv $BIND --env PYTHONPATH="$OUT/pypath" "$SIF" python3 -c "
import trtllm_rs, trtllm_rs_bridge
s = trtllm_rs.RustScheduler(64, 8192, 20.0, 1000)
decide_args = (
    [1,2],             # ids
    [True,False],      # is_context
    [10,10],           # prompt_lens
    [0,5],             # context_done_tokens
    [0,5],             # tokens_generated
    [100,100],         # max_new_tokens
    [0.0,0.0],         # arrival_ms
    [],                # inflight
    500,               # kv_free_blocks
    1.0,               # now_ms
)
assert len(decide_args) == 10, f'RustScheduler.decide ABI changed: {len(decide_args)} args'
assert decide_args[3] == [0,5], 'context_done_tokens must be the fourth ABI argument'
print('  import ok:', s.decide(*decide_args))
" || { echo "!!! import gate failed"; exit 11; }

# mpirun is mandatory even at TP1: MpiPoolSession uses MPI_Comm_spawn and a bare launch dies with
# MPI_ERR_SPAWN (job 311967).
apptainer exec --nv $BIND \
  --env PYTHONPATH="$OUT/pypath" \
  --env TRTLLM_RS_MODE="$MODE" \
  --env TRTLLM_RS_REPORT="$OUT/rs-shadow.json" \
  "$SIF" \
  mpirun --bind-to none --oversubscribe -H "$(hostname):12" -n 1 \
  python3 "$REPO/scripts/shadow_launch.py" serve "$MODEL" \
    --backend pytorch --tp_size 1 --host 0.0.0.0 --port "$PORT" --trust_remote_code \
    --kv_cache_free_gpu_memory_fraction 0.80 \
    --extra_llm_api_options "$OUT/engine.yaml" > "$OUT/server.log" 2>&1 &
SPID=$!

ready=0
for i in $(seq 1 72); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ] \
    && { ready=1; echo "  ready at t=$((i*5))s"; break; }
  kill -0 "$SPID" 2>/dev/null || { echo "!!! SERVER DIED"; tail -50 "$OUT/server.log"; exit 20; }
  sleep 5
done
[ "$ready" = 1 ] || { echo "!!! NOT READY"; tail -60 "$OUT/server.log"; kill -- -"$SPID" 2>/dev/null; exit 21; }

# The scheduler must have actually been swapped. A silently un-patched run looks exactly like
# "Rust agreed with everything" -- the most dangerous possible outcome for a shadow gate.
grep -aq "\[trtllm-rs\] RustScheduler(" "$OUT/server.log" \
  || { echo "!!! the Rust scheduler was NEVER INSTALLED -- any agreement below would be vacuous";
       grep -aiE "trtllm-rs|traceback|importerror" "$OUT/server.log" | tail -20;
       kill -- -"$SPID" 2>/dev/null; exit 22; }
echo "  installed: $(grep -a '\[trtllm-rs\] RustScheduler(' "$OUT/server.log" | tail -1)"

MID=$(curl -s --max-time 8 "http://127.0.0.1:$PORT/v1/models" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
[ -n "$MID" ] || { echo "!!! /v1/models empty"; kill -- -"$SPID" 2>/dev/null; exit 23; }

# Same aiperf line as A0/A1/A2 so the numbers stay comparable to the baselines.
"$AIPERF" profile --model "$MID" --tokenizer "$MODEL" \
  --url "http://127.0.0.1:$PORT" --endpoint-type chat --endpoint /v1/chat/completions \
  --streaming --concurrency 32 --workers-max 32 --request-count 320 --warmup-request-count 32 \
  --synthetic-input-tokens-mean 512 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 128 --output-tokens-stddev 0 \
  --extra-inputs min_tokens:128 --extra-inputs max_tokens:128 --extra-inputs ignore_eos:true \
  --extra-inputs temperature:0.0 --random-seed 2026 --ui simple --use-legacy-max-tokens \
  --output-artifact-dir "$OUT/load" > "$OUT/aiperf.log" 2>&1
echo "  aiperf exit=$?"
sed 's/\x1b\[[0-9;]*m//g' "$OUT/aiperf.log" | grep -iE "Inter Token Latency|Time to First Token|Request Throughput|Output Token Throughput" | sed 's/^/    /'

# SIGTERM, not SIGKILL: the launcher writes the comparison from its signal handler.
echo; echo "  stopping server"
kill -TERM -- -"$SPID" 2>/dev/null
for i in $(seq 1 18); do kill -0 "$SPID" 2>/dev/null || break; sleep 5; done
kill -KILL -- -"$SPID" 2>/dev/null; sleep 3

echo; echo "########## SHADOW COMPARISON ##########"
if [ -s "$OUT/rs-shadow.json" ]; then
  python3 -c "
import json,sys
d=json.load(open('$OUT/rs-shadow.json'))
print('  mode:', d.get('mode'))
for i,e in enumerate(d.get('executors',[])):
    print(f'  executor {i}: {e.get(\"summary\")}')
    if e.get('rust_stats'): print(f'    rust: {e[\"rust_stats\"]}')
    for k,v in list(enumerate(e.get('divergences',[])))[:5]:
        print(f'    divergence {k}: {str(v)[:200]}')
"
else
  echo "  !!! no report written"; grep -a "trtllm-rs" "$OUT/server.log" | tail -10
fi
echo "=== done $(date -Is) ==="; echo "artifacts: $OUT"
