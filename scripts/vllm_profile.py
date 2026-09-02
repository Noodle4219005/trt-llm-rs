"""vLLM's prefill and decode capacity on this hardware, measured by us.

The number this project has been comparing itself against -- 12,050 prefill
tok/s/GPU -- is inferred from a teammate's 22.419 goodput, not measured. Every
conclusion that rests on it deserves a direct measurement, and the same harness
that profiled TensorRT-LLM (jobs 328143, 328182) makes it a like-for-like
comparison rather than two numbers from different methods.
"""

import json
import os
import time

from vllm import LLM, SamplingParams

MODEL = os.environ["MODEL"]
ROLE = os.environ.get("PROF_ROLE", "prefill")
TP = int(os.environ.get("PROF_TP", "2"))
CHUNK = int(os.environ.get("PROF_CHUNK", "0"))
# Tokens the engine may schedule in one step, and requests per generate() call.
# The first measurement used 16384 and a batch of 4 -- 16000 tokens, right at
# the cap -- and produced 9,124 tok/s/GPU, below what a teammate's 22.419
# goodput implies 8 prefill GPUs must have sustained. Either the batch shape
# understates the engine or that goodput had help; this makes the shape a
# variable so the first can be ruled out.
BUDGET = int(os.environ.get("PROF_BUDGET", "16384"))
BATCH = int(os.environ.get("PROF_BATCH", "4"))
MAXSEQ = int(os.environ.get("PROF_MAXSEQ", "256"))
# Expert parallelism changes which MoE kernel runs, not just how work is split.
# On Hopper with a [128,128] block-FP8 checkpoint, oracle/fp8.py:112-120 keeps
# Triton for TP and moves FlashInfer CUTLASS to the front for EP -- and the
# CUTLASS path quantises activations inside the kernel
# (flashinfer_cutlass_moe.py: expects_unquantized_inputs=True) rather than in a
# separate pass. That separate pass is 30.86% of prefill wall clock on
# TensorRT-LLM, so this is the one configuration lever that could remove it.
EP = os.environ.get("PROF_EP", "0") == "1"
ISL = 4000


def main() -> None:
    kw = dict(
        model=MODEL,
        tensor_parallel_size=TP,
        max_model_len=4608,
        gpu_memory_utilization=0.90,
        max_num_seqs=MAXSEQ,
        enable_expert_parallel=EP,
        enforce_eager=False,
        # The gate is cold-cache, and prefix reuse would measure the cache.
        enable_prefix_caching=False,
        kv_cache_dtype="fp8",
    )
    # 0.27.1 gates profiling behind ProfilerConfig, not the env var alone:
    # "Profiling is not enabled. Please set --profiler-config".
    tdir = os.environ.get("VLLM_TORCH_PROFILER_DIR")
    if tdir:
        kw["profiler_config"] = {"profiler": "torch", "torch_profiler_dir": tdir}
    if CHUNK > 0:
        kw["max_num_batched_tokens"] = CHUNK
        kw["enable_chunked_prefill"] = True
    else:
        kw["max_num_batched_tokens"] = BUDGET
    llm = LLM(**kw)
    out = {"role": ROLE, "tp": TP, "chunk": CHUNK, "ep": EP}

    prompt = {"prompt_token_ids": [1234] * ISL}

    if ROLE in ("prefill", "both"):
        sp1 = SamplingParams(max_tokens=1, temperature=0.0)
        for _ in range(2):
            llm.generate([prompt] * BATCH, sp1)
        rounds, batch = 10, BATCH
        t0 = time.perf_counter()
        for _ in range(rounds):
            llm.generate([prompt] * batch, sp1)
        dt = time.perf_counter() - t0
        toks = rounds * batch * ISL
        out["prefill"] = {
            "budget": BUDGET,
            "ep": EP,
            "batch": batch,
            "seconds": round(dt, 3),
            "tok_s_total": round(toks / dt, 1),
            "tok_s_per_gpu": round(toks / dt / TP, 1),
            "req_s": round(rounds * batch / dt, 3),
        }
        print("PREFILL " + json.dumps(out["prefill"]), flush=True)

        # A trace of the same shape that was just timed. TensorRT-LLM's
        # equivalent showed 30.86% of prefill in FP8 activation quantisation
        # and only 48.7% in real compute; vLLM is 1.34-1.55x faster and the
        # remaining 2.3x to a 70 req/s target has to come from somewhere this
        # names.
        if os.environ.get("VLLM_TORCH_PROFILER_DIR"):
            llm.start_profile()
            for _ in range(3):
                llm.generate([prompt] * BATCH, sp1)
            llm.stop_profile()
            print("PROFILED", flush=True)

    if ROLE in ("decode", "both"):
        short = {"prompt_token_ids": [1234] * 128}
        sp200 = SamplingParams(max_tokens=200, temperature=0.0, ignore_eos=True)
        out["decode"] = []
        for conc in (32, 64, 128, 192):
            llm.generate([short] * 4, SamplingParams(max_tokens=8, temperature=0.0))
            t0 = time.perf_counter()
            res = llm.generate([short] * conc, sp200)
            dt = time.perf_counter() - t0
            got = sum(len(r.outputs[0].token_ids) for r in res)
            row = {
                "concurrency": conc,
                "seconds": round(dt, 3),
                "itl_ms": round(dt * 1000.0 / max(1.0, got / conc - 1.0), 3),
                "tok_s_per_gpu": round(got / dt / TP, 1),
            }
            out["decode"].append(row)
            print("DECODE " + json.dumps(row), flush=True)

    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
