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
ISL = 4000


def main() -> None:
    kw = dict(
        model=MODEL,
        tensor_parallel_size=TP,
        max_model_len=4608,
        gpu_memory_utilization=0.90,
        enforce_eager=False,
        # The gate is cold-cache, and prefix reuse would measure the cache.
        enable_prefix_caching=False,
        kv_cache_dtype="fp8",
    )
    if CHUNK > 0:
        kw["max_num_batched_tokens"] = CHUNK
        kw["enable_chunked_prefill"] = True
    else:
        kw["max_num_batched_tokens"] = 16384
    llm = LLM(**kw)
    out = {"role": ROLE, "tp": TP, "chunk": CHUNK}

    prompt = {"prompt_token_ids": [1234] * ISL}

    if ROLE in ("prefill", "both"):
        sp1 = SamplingParams(max_tokens=1, temperature=0.0)
        for _ in range(2):
            llm.generate([prompt] * 4, sp1)
        rounds, batch = 10, 4
        t0 = time.perf_counter()
        for _ in range(rounds):
            llm.generate([prompt] * batch, sp1)
        dt = time.perf_counter() - t0
        toks = rounds * batch * ISL
        out["prefill"] = {
            "seconds": round(dt, 3),
            "tok_s_total": round(toks / dt, 1),
            "tok_s_per_gpu": round(toks / dt / TP, 1),
            "req_s": round(rounds * batch / dt, 3),
        }
        print("PREFILL " + json.dumps(out["prefill"]), flush=True)

    if ROLE in ("decode", "both"):
        short = {"prompt_token_ids": [1234] * 128}
        sp200 = SamplingParams(max_tokens=200, temperature=0.0, ignore_eos=True)
        out["decode"] = []
        for conc in (16, 32, 64):
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
