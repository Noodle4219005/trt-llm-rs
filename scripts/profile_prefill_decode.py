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
# Chunked-prefill token budget per iteration; 0 disables chunking.
CHUNK = int(os.environ.get("PROF_CHUNK", "0"))
ISL = 4000


def main() -> None:
    # TP2 is the prefill shape this deployment runs. Decode runs TP4 in
    # production, so the decode rows below are reported per GPU and scale from
    # there rather than being read as a TP4 measurement.
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=TP,
        moe_expert_parallel_size=1,
        max_seq_len=4608,
        max_batch_size=64,
        attn_backend="TRTLLM",
        moe_config=MoeConfig(backend="AUTO"),
        kv_cache_config=KvCacheConfig(
            dtype="fp8", enable_block_reuse=False, free_gpu_memory_fraction=0.75
        ),
        # Chunked prefill is what actually interleaves the two phases. Job
        # 328269 measured in-card P/D with it OFF and got -6% overlap: the
        # engine ran prefill then decode serially and the whole +27% over
        # disaggregation came from removing the pool boundary, not from
        # overlapping. With it on, a long prompt is split so decode steps run
        # between the pieces -- which is the mechanism, and the knob a
        # scheduler steers.
        enable_chunked_prefill=CHUNK > 0,
        max_num_tokens=CHUNK if CHUNK > 0 else 16384,
        disable_overlap_scheduler=True,  # context servers, per the disagg docs
    )
    out = {"role": ROLE, "tp": TP, "chunk": CHUNK}

    if ROLE == "mixed":
        _run_mixed(llm, out)
    if ROLE in ("prefill", "both"):
        _run_prefill(llm, out)
    if ROLE in ("decode", "both"):
        _run_decode(llm, out)
    print("RESULT " + json.dumps(out), flush=True)
    llm.shutdown()


def _run_mixed(llm, out) -> None:
    """In-card P/D: one worker doing both, which is the only way to hit a
    fractional prefill:decode split.

    The optimal split moves as prefill gets faster -- 10.9/5.1 GPUs at our
    measured 15.1% prefill MFU, 5.7/10.3 at the 59.3% the 38 req/s target
    implies -- so a physical pool split has to be re-cut every time the kernels
    improve, while a shared card only changes a token-budget ratio. It also
    removes the KV transfer entirely: the cache is already where decode needs
    it.

    Prediction, from per-request GPU-seconds at the measured rates
    (4000/6804 + 200/726 = 0.863 s): 16 GPUs perfectly packed give 18.53 req/s,
    so one TP2 replica of eight should give 2.32.
    """
    prompt = [1234] * ISL
    greedy = SamplingParams(max_tokens=200, temperature=0.0, ignore_eos=True)
    out["mixed"] = []
    for conc in (8, 16, 24):
        llm.generate([[1234] * 128] * 2, SamplingParams(max_tokens=4, temperature=0.0))
        t0 = time.perf_counter()
        res = llm.generate([prompt] * conc, greedy)
        dt = time.perf_counter() - t0
        got = sum(len(r.outputs[0].token_ids) for r in res)
        row = {
            "concurrency": conc,
            "seconds": round(dt, 3),
            "out_tokens": got,
            "req_s": round(conc / dt, 3),
            "req_s_per_gpu": round(conc / dt / TP, 4),
            # What sixteen GPUs of this shape would sustain.
            "req_s_at_16gpu": round(conc / dt / TP * 16, 2),
        }
        out["mixed"].append(row)
        print("MIXED " + json.dumps(row), flush=True)


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
