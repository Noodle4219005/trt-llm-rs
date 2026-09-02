#!/bin/bash
#SBATCH -J profile-prefill
#SBATCH -p 8gpus
#SBATCH -N 1
#SBATCH --gres=gpu:H200:2
#SBATCH -t 00:20:00
#SBATCH -o /work/u5727520/trt-llm-rs-runs/logs/%x-%j.out
#SBATCH -e /work/u5727520/trt-llm-rs-runs/logs/%x-%j.err
#
# Where does prefill's time go?
#
# goodput 25 needs 26.9 req/s and we measure 13.74, all of it prefill-bound:
# 6,870 tok/s/GPU, 15.3% MFU against a vLLM stack reaching 12,050 and 26.9% on
# the same hardware. Every other constraint is settled -- ITL 15.3 ms with
# speculation against a 20 ms gate, TTFT 1.6 s against 3000, and the P/D ratio
# swept over 88 points. This is the only number left, and it has never been
# profiled on THIS stack: the 25.2%-unoverlapped-AllReduce figure this project
# has been reasoning from was measured on SGLang.
#
# Two GPUs rather than sixteen: one TP2 prefill worker is the shape we run, and
# 2 SU/min instead of 16 buys the whole answer for about 25 SU.
set -uo pipefail
SRC=/home/u5727520/trt-llm-rs
SIF=/work/u5727520/dynamo-qwen-smoke/images/tensorrtllm-runtime-1.4.1.sif
MODEL=/work/u4063814/hf_cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969
OUT=/work/u5727520/trt-llm-rs-runs/out/profile-$SLURM_JOB_ID
BIND="--bind /work/u4063814,/work/u5727520,/home/u5727520"
mkdir -p "$OUT"

AVAIL_GB=$(df -BG --output=avail /work/u5727520 2>/dev/null | tail -1 | tr -dc '0-9')
echo "/work available: ${AVAIL_GB:-unknown} GiB"
[ -n "${AVAIL_GB:-}" ] && [ "$AVAIL_GB" -lt 3 ] && { echo "!!! too little disk"; exit 29; }

# The mmap loader: at TP2 each rank holds 110 GiB of weights, and without a
# shared read-only mapping two ranks on one node charge the page cache twice.
cp "$SRC/scripts/usercustomize.py" "$SRC/scripts/trtllm_rs_mmap_weights.py" "$OUT/"

cat > "$OUT/prof.py" <<'PY'
import os, time, json
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig

MODEL = os.environ["MODEL"]
ISL, BATCH = 4000, 4          # 4 x 4000 = 16000, just under max_num_tokens
llm = LLM(
    model=MODEL,
    tensor_parallel_size=2,
    moe_expert_parallel_size=1,
    max_num_tokens=16384,
    max_seq_len=4608,
    max_batch_size=8,
    attn_backend="TRTLLM",
    moe_config=MoeConfig(backend="AUTO"),
    kv_cache_config=KvCacheConfig(dtype="fp8", enable_block_reuse=False,
                                  free_gpu_memory_fraction=0.70),
    disable_overlap_scheduler=True,   # context servers, per the disagg docs
)
# max_tokens=1 keeps every iteration a context iteration.
sp = SamplingParams(max_tokens=1, temperature=0.0)
prompt = [1234] * ISL

print("warmup", flush=True)
for _ in range(3):
    llm.generate([prompt] * BATCH, sp)

print("timed", flush=True)
t0 = time.perf_counter()
ROUNDS = 12
for _ in range(ROUNDS):
    llm.generate([prompt] * BATCH, sp)
dt = time.perf_counter() - t0
toks = ROUNDS * BATCH * ISL
print(json.dumps({
    "rounds": ROUNDS, "batch": BATCH, "isl": ISL,
    "seconds": round(dt, 3),
    "tok_s_total": round(toks / dt, 1),
    "tok_s_per_gpu": round(toks / dt / 2, 1),
    "req_s": round(ROUNDS * BATCH / dt, 3),
}), flush=True)
llm.shutdown()
PY

echo "=== profiling one TP2 prefill worker ==="
srun -N1 -n1 --gres=gpu:H200:2 apptainer exec --nv $BIND \
  --env "MODEL=$MODEL" --env "PYTHONPATH=$OUT" \
  --env "TLLM_PROFILE_START_STOP=8-12" \
  --env "TLLM_TORCH_PROFILE_TRACE=$OUT/prefill-trace.json" \
  --env "TRTLLM_ENABLE_PDL=1" \
  "$SIF" python3 "$OUT/prof.py" 2>&1 | tail -40

echo "=== artifacts ==="
ls -la "$OUT"/*.json 2>/dev/null | awk '{printf "  %s  %.1f MB\n", $9, $5/1048576}'
