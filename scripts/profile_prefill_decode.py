"""Measure one worker's prefill and decode capacity, with a torch trace.

Two numbers decide everything left in this project.

Prefill: goodput 25 needs 26.9 req/s and we measure 13.74, all of it
prefill-bound at 6,870 tok/s/GPU and 15.3% MFU, against a vLLM stack reaching
12,050 on the same hardware. The 25.2%-unoverlapped-AllReduce figure this
project has reasoned from was measured on SGLang, so this is the first time the
question is asked of our own stack.

Decode: every disaggregated run so far was prefill-bound, so the decode side
never reached its own ceiling and its capacity has never been measured. That
number is also exactly what decides whether a vLLM-prefill + TRT-LLM-decode
hybrid is worth building: it is worth it only if this side can sustain the
target rate on its own.
"""

import json
import os
import time

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig

MODEL = os.environ["MODEL"]
ROLE = os.environ.get("PROF_ROLE", "both")   # prefill | decode | both
TP = int(os.environ.get("PROF_TP", "2"))
ISL = 4000


def main() -> None:
    # TP2 is the prefill shape this deployment runs. Decode runs TP4 in
    # production, so the decode rows below are reported per GPU and scale from
    # there rather than being read as a TP4 measurement.
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=TP,
        moe_expert_parallel_size=1,
        max_num_tokens=16384,
        max_seq_len=4608,
        max_batch_size=64,
        attn_backend="TRTLLM",
        moe_config=MoeConfig(backend="AUTO"),
        kv_cache_config=KvCacheConfig(
            dtype="fp8", enable_block_reuse=False, free_gpu_memory_fraction=0.75
        ),
        disable_overlap_scheduler=True,  # context servers, per the disagg docs
    )
    out = {"role": ROLE, "tp": TP}

    if ROLE in ("prefill", "both"):
        _run_prefill(llm, out)
    if ROLE in ("decode", "both"):
        _run_decode(llm, out)
    print("RESULT " + json.dumps(out), flush=True)
    llm.shutdown()


def _run_prefill(llm, out) -> None:
    """max_tokens=1 keeps every iteration a context iteration, so the timing is
    prefill and nothing else."""
    greedy1 = SamplingParams(max_tokens=1, temperature=0.0)
    prompt = [1234] * ISL
    batch = 4  # 4 x 4000 = 16000, just under max_num_tokens
    print("prefill warmup", flush=True)
    for _ in range(2):
        llm.generate([prompt] * batch, greedy1)

    print("prefill timed", flush=True)
    rounds = 10
    t0 = time.perf_counter()
    for _ in range(rounds):
        llm.generate([prompt] * batch, greedy1)
    dt = time.perf_counter() - t0
    toks = rounds * batch * ISL
    out["prefill"] = {
        "rounds": rounds,
        "batch": batch,
        "seconds": round(dt, 3),
        "tok_s_total": round(toks / dt, 1),
        "tok_s_per_gpu": round(toks / dt / TP, 1),
        "req_s": round(rounds * batch / dt, 3),
    }
    print("PREFILL " + json.dumps(out["prefill"]), flush=True)


def _run_decode(llm, out) -> None:

    # Phase B -- decode capacity, saturated. A short prompt so the 200 output
    # tokens dominate the measurement.
    short = [1234] * 128
    greedy200 = SamplingParams(max_tokens=200, temperature=0.0, ignore_eos=True)
    out["decode"] = []
    for conc in (16, 32, 48, 64):
        llm.generate([short] * 4, SamplingParams(max_tokens=8, temperature=0.0))
        t0 = time.perf_counter()
        res = llm.generate([short] * conc, greedy200)
        dt = time.perf_counter() - t0
        got = sum(len(r.outputs[0].token_ids) for r in res)
        per_seq = got / conc
        row = {
            "concurrency": conc,
            "seconds": round(dt, 3),
            "tokens": got,
            # Wall clock over the gaps one sequence actually produced.
            "itl_ms": round(dt * 1000.0 / max(1.0, per_seq - 1.0), 3),
            "tok_s_per_gpu": round(got / dt / TP, 1),
            "req_s": round(conc / dt, 3),
        }
        out["decode"].append(row)
        print("DECODE " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
