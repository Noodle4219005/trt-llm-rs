#!/usr/bin/env python3
"""Run trtllm-serve with the Rust scheduler installed, and report what it decided.

TensorRT-LLM's executor loop is left completely alone. The only thing that changes is the object
behind `PyExecutor.scheduler.capacity_scheduler`, which decides admission -- which requests are
allowed to run this step. Everything downstream of it stays upstream code: the micro-batch
scheduler still owns chunked-prefill chunk sizing, the token budget, encoder requests and the
context/generation split, and the loop still owns speculative decoding, CUDA graphs, sampling, KV
and disaggregation. See ADR 0034.

The capacity layer is the right seam because it is exactly where the measured headroom is: 53
sequences at a mean ITL of 17.23 ms leaves 14% of a 20 ms budget unspent, and that is a decision
about who gets admitted, not about how the batch is then assembled.

Default mode is `shadow`: both schedulers decide, the divergences are recorded, and THE GPU EATS
THE UPSTREAM DECISION. Nothing about the run's behaviour changes; the only output is a comparison.
That makes it free to attach to any job, which is the point -- it is the cheapest correctness gate
in the project.

    TRTLLM_RS_MODE=shadow|live      default shadow
    TRTLLM_RS_REPORT=<path>         where to write the comparison (default $PWD/rs-shadow.json)

Usage mirrors trtllm-serve exactly:
    python3 shadow_launch.py serve <model> --backend pytorch --tp_size 1 ...
"""
import atexit
import json
import os
import signal
import sys

MODE = os.environ.get("TRTLLM_RS_MODE", "shadow")
REPORT = os.environ.get("TRTLLM_RS_REPORT", os.path.join(os.getcwd(), "rs-shadow.json"))

import trtllm_rs  # the Rust extension module (cdylib)
import trtllm_rs_bridge.patch as patch


def _factory(py_executor):
    """Size the Rust scheduler from the executor that is asking for it.

    These numbers are only knowable after TensorRT-LLM has resolved its own config, which is why
    install_global takes a factory rather than a prebuilt instance.
    """
    args = getattr(py_executor, "llm_args", None) or getattr(py_executor, "_llm_args", None)
    def pick(*names, default=None):
        for src in (py_executor, args):
            for n in names:
                v = getattr(src, n, None) if src is not None else None
                if v:
                    return v
        return default

    max_batch = int(pick("max_batch_size", "max_num_sequences", default=256))
    max_tokens = int(pick("max_num_tokens", default=8192))
    kv_blocks = 0
    rm = getattr(py_executor, "resource_manager", None)
    for attr in ("get_max_resource_count", "get_total_blocks", "num_blocks"):
        fn = getattr(rm, attr, None) if rm is not None else None
        if callable(fn):
            try:
                kv_blocks = int(fn())
                break
            except Exception:
                pass
    print(f"[trtllm-rs] RustScheduler(max_batch={max_batch}, max_tokens={max_tokens}, "
          f"kv_blocks={kv_blocks}, mode={MODE})", flush=True)
    return trtllm_rs.RustScheduler(max_batch, max_tokens, 20.0, kv_blocks)


_handle = patch.install_global(_factory, mode=MODE)


def _dump(*_a):
    """Write the comparison out. Registered for both normal exit and SIGTERM.

    A shadow run whose result is never written is a run that cost GPU time and taught nothing, and
    a server is almost always killed rather than allowed to return -- so atexit alone is not
    enough.
    """
    try:
        out = {"mode": MODE, "executors": []}
        for ex in patch.wrapped_executors():
            # The swap happens one level down, on scheduler.capacity_scheduler.
            outer = getattr(ex, "scheduler", None)
            sched = getattr(outer, "capacity_scheduler", None)
            if sched is None or not hasattr(sched, "summary"):
                continue
            entry = {"summary": sched.summary()}
            if hasattr(sched, "divergences"):
                entry["divergences"] = sched.divergences()
            rust = getattr(sched, "rust", None)
            if rust is not None and hasattr(rust, "stats"):
                entry["rust_stats"] = rust.stats()
                entry["observer_samples"] = entry["rust_stats"].get("samples", 0)
            out["executors"].append(entry)
        with open(REPORT, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"[trtllm-rs] wrote {REPORT}", flush=True)
        for e in out["executors"]:
            print(f"[trtllm-rs] {e['summary']}", flush=True)
    except Exception as exc:  # never let reporting break the shutdown path
        print(f"[trtllm-rs] failed to write report: {exc}", flush=True)


atexit.register(_dump)
for _sig in (signal.SIGTERM, signal.SIGINT):
    _prev = signal.getsignal(_sig)
    def _handler(signum, frame, _prev=_prev):
        _dump()
        if callable(_prev):
            _prev(signum, frame)
        else:
            sys.exit(0)
    signal.signal(_sig, _handler)

from tensorrt_llm.commands.serve import main  # noqa: E402  (must come after the patch)

if __name__ == "__main__":
    sys.exit(main())
