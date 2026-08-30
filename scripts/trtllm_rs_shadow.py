#!/usr/bin/env python3
"""Shadow the TensorRT-LLM capacity scheduler with the Rust one, and count.

ADR 0034 fixes the hook point: Rust owns *policy* by standing in for the
`capacity_scheduler`, and TensorRT-LLM's executor loop is not rewritten. This
module is the shadow stage of that decision -- both schedulers run every step,
**the GPU only ever executes upstream's decision**, and we count how far the
Rust policy would have diverged.

What the counters mean (ADR 0034's Update 2026-08-29 corrected this, and it is
the whole reason this file exists):

    diverged                 EXPECTED SIGNAL, not failure. Upstream implements
                             GuaranteedNoEvict; ours is an ITL-budget admission
                             controller. They are *meant* to differ -- the
                             project exists because 20 ms of ITL budget is 14%
                             unspent. A run reporting zero divergence is the
                             suspicious one: the most likely explanation is that
                             this shim never got installed, and "Rust agrees
                             perfectly" is indistinguishable from "the patch did
                             nothing" in a report.
    malformed                REAL failure. Index out of range, sets not
                             disjoint, or a request scheduled that upstream
                             could not schedule at all.
    empty_when_work_available REAL failure. Rust admitted nothing while
                             candidates existed.

Installed via `usercustomize` so it also reaches the MPI-spawned executor
processes, which do not inherit a parent's monkeypatch. Set TRTLLM_RS_SHADOW=1
to arm it; anything else and this module is inert.

The shim must never be able to take the run down: every read of an upstream
request object is defensive, and after TRTLLM_RS_SHADOW_MAX_ERRORS failures it
disables itself and lets the engine continue on upstream's decisions alone.
"""

from __future__ import annotations

import atexit
import json
import os
import random
import sys
import time
from typing import Any

_MAX_ERRORS = int(os.environ.get("TRTLLM_RS_SHADOW_MAX_ERRORS", "20"))
# Dump every N steps, not only at exit. Job 314766 armed correctly, ran the full
# 320-request workload, and still produced no report: the executor lives in an
# MPI_Comm_spawn'd process that is torn down without running atexit handlers.
# A counter that only exists in the memory of a process nobody shuts down
# gracefully is a counter you do not have.
_DUMP_EVERY = int(os.environ.get("TRTLLM_RS_SHADOW_DUMP_EVERY", "100"))
# Shadow by default. TRTLLM_RS_LIVE=1 makes the engine execute the Rust
# decision instead of upstream's -- the whole point of the project, and the one
# switch that can make a run produce wrong output rather than merely a wrong
# number. It is opt-in, and even then a malformed decision falls back to
# upstream for that step rather than being executed.
_LIVE = os.environ.get("TRTLLM_RS_LIVE") == "1"


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(q * len(ordered)))
    return round(ordered[idx], 3)


class ShadowState:
    """Counters plus the arrival clock TensorRT-LLM does not expose.

    rc22's LlmRequest has no readable arrival time: `arrival_time` is a
    write-only ctor kwarg that lands in mPerfMetrics and is never re-exposed.
    First-sight time here is the honest substitute, and it is recorded as such.
    """

    def __init__(self) -> None:
        self.steps = 0
        self.diverged = 0
        self.malformed = 0
        self.empty_when_work_available = 0
        self.errors = 0
        self.disabled = False
        self.upstream_admitted = 0
        self.rust_admitted = 0
        self.live = _LIVE
        self.live_steps = 0
        self.live_fallbacks = 0
        self.candidates_seen = 0
        self.first_seen_ms: dict[int, float] = {}
        self.last_call_ms: float | None = None
        self.observe_calls = 0
        # rid -> (time we last saw it, tokens it had generated then). This is
        # what makes a *true* inter-token latency measurable from this seam.
        self.token_progress: dict[int, tuple[float, int]] = {}
        # The old proxy, kept only so the two can be compared in one artifact.
        self.iteration_ms_ewma = 0.0
        # Steering signal: p90 across live started requests of their projected
        # per-request mean ITL. This is the quantity the SLO is written in.
        self.steer_itl_ms_ewma = 0.0
        # Reporting only: mean interval of sequences that advanced this step.
        # Was called true_itl_ms_ewma, which was a lie -- it is true only when
        # every resident sequence advances every step.
        self.advancing_itl_ms_ewma = 0.0

        # A mean that equals the iteration gap to four decimals is not evidence
        # that sequences advance once per iteration -- it is equally consistent
        # with the token counter never moving. Keep the raw shape.
        self.generating_seen = 0
        self.advanced_seen = 0
        # Uniform reservoirs, NOT prefixes. The previous version appended while
        # `len < 20000`, which keeps the FIRST 20,000 samples and drops every
        # one after -- at 25 samples per step that is the first ~800 steps, so
        # the percentiles described the warmup ramp and were labelled as if they
        # described the run. In job 316849 that read p50 = 93.77 ms while the
        # steady state was ~15 ms. The comment on it said "Reservoir-ish".
        self.advancing_itl: list[float] = []
        self.steer_itl: list[float] = []
        # One-element lists so the counter can be bumped in place by whoever
        # holds the matching pool, without naming it again at every call.
        self.advancing_itl_count = [0]
        self.steer_itl_count = [0]
        self.reason_first_error = ""

    def reservoir(
        self,
        pool: list[float],
        counter: list[int],
        sample: float,
        cap: int = 20000,
    ) -> None:
        """Algorithm R: every sample seen keeps an equal chance of being held.

        The version this replaces appended while `len(pool) < cap`, which is a
        prefix, not a reservoir: it keeps the first `cap` samples and discards
        every later one. At ~25 samples per scheduler step that is the first
        ~800 steps, so the percentiles reported the warmup ramp under a name
        that reads like the whole run.
        """
        # Counts live beside their pool rather than behind a formatted
        # attribute name: this runs once per running request per scheduler
        # step, ~74 times every 15 ms, in a control plane whose whole budget is
        # a few percent of the iteration.
        counter[0] += 1
        seen = counter[0]
        if len(pool) < cap:
            pool.append(sample)
            return
        j = random.randrange(seen)
        if j < cap:
            pool[j] = sample

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "diverged": self.diverged,
            "malformed": self.malformed,
            "empty_when_work_available": self.empty_when_work_available,
            "errors": self.errors,
            "disabled": self.disabled,
            "upstream_admitted": self.upstream_admitted,
            "rust_admitted": self.rust_admitted,
            "candidates_seen": self.candidates_seen,
            "observe_calls": self.observe_calls,
            "iteration_ms_ewma": round(self.iteration_ms_ewma, 4),
            "steer_itl_ms_ewma": round(self.steer_itl_ms_ewma, 4),
            "advancing_itl_ms_ewma": round(self.advancing_itl_ms_ewma, 4),
            "generating_seen": self.generating_seen,
            "advanced_seen": self.advanced_seen,
            "advance_fraction": round(
                self.advanced_seen / self.generating_seen, 4
            ) if self.generating_seen else None,
            # What the controller steers on: time each running request has
            # waited for its next token. This is the SLO-shaped one.
            "steer_itl_p50": _pct(self.steer_itl, 0.50),
            "steer_itl_p90": _pct(self.steer_itl, 0.90),
            "steer_itl_p99": _pct(self.steer_itl, 0.99),
            "steer_itl_seen": self.steer_itl_count[0],
            # Interval between tokens of sequences the engine IS advancing.
            # Says how fast the engine is, NOT how long requests wait -- the
            # two differ by exactly the starvation this controller looks for.
            "advancing_itl_p50": _pct(self.advancing_itl, 0.50),
            "advancing_itl_p90": _pct(self.advancing_itl, 0.90),
            "advancing_itl_p99": _pct(self.advancing_itl, 0.99),
            "advancing_itl_seen": self.advancing_itl_count[0],
            "live": self.live,
            "live_steps": self.live_steps,
            "live_fallbacks": self.live_fallbacks,
            "reason_first_error": self.reason_first_error,
        }


STATE = ShadowState()


def _dump() -> None:
    path = os.environ.get("TRTLLM_RS_SHADOW_OUT")
    if not path:
        return
    payload = STATE.as_dict()
    instance = _LIVE_INSTANCE
    if instance is not None:
        try:
            payload["rust_stats"] = instance.rust_stats()
        except Exception:  # pragma: no cover - a report must never raise
            pass
    try:
        with open(f"{path}.{os.getpid()}", "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    except Exception:  # pragma: no cover - never fail the run over a report
        pass


_INSTALLED: Any = None
_LIVE_INSTANCE: Any = None


def install() -> bool:
    """Replace `_util.BindCapacityScheduler` with the shadowing subclass.

    Returns True when the patch is in place. `_util` is the only module that
    constructs the capacity scheduler (`_util.py:2433`), so patching the name in
    that namespace is enough and nothing under third_party/ is edited.
    """
    global _INSTALLED

    try:
        from tensorrt_llm._torch.pyexecutor import _util
        from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import (
            BindCapacityScheduler,
        )
    except Exception as exc:
        print(f"[trtllm-rs-shadow] not installing: {exc}", file=sys.stderr, flush=True)
        return False

    try:
        from trtllm_rs import RustScheduler
    except Exception as exc:
        print(f"[trtllm-rs-shadow] trtllm_rs unavailable: {exc}", file=sys.stderr, flush=True)
        return False

    itl_budget_ms = float(os.environ.get("TRTLLM_RS_ITL_BUDGET_MS", "20.0"))

    class ShadowCapacityScheduler(BindCapacityScheduler):
        def __init__(self, max_num_requests: int, kv_cache_manager: Any, *args: Any, **kwargs: Any) -> None:
            super().__init__(max_num_requests, kv_cache_manager, *args, **kwargs)
            self._rust = RustScheduler(
                max_num_requests,
                int(os.environ.get("TRTLLM_RS_MAX_NUM_TOKENS", "8192")),
                itl_budget_ms,
                self._total_kv_blocks(),
            )
            global _LIVE_INSTANCE
            _LIVE_INSTANCE = self
            print(
                "[trtllm-rs-shadow] armed: max_num_requests="
                f"{max_num_requests} itl_budget_ms={itl_budget_ms}",
                file=sys.stderr,
                flush=True,
            )

        def _total_kv_blocks(self) -> int:
            for name in ("get_max_capacity_batch_size", "get_num_free_blocks"):
                value = getattr(self.kv_cache_manager, name, None)
                if callable(value):
                    try:
                        return int(value())
                    except Exception:
                        continue
            return 0

        def _free_blocks(self) -> int:
            getter = getattr(self.kv_cache_manager, "get_num_free_blocks", None)
            if callable(getter):
                try:
                    return int(getter())
                except Exception:
                    return 0
            return 0

        def rust_stats(self) -> dict[str, float]:
            try:
                return dict(self._rust.stats())
            except Exception:
                return {}

        def schedule_request(self, active_requests: Any):
            upstream = super().schedule_request(active_requests)
            if STATE.disabled:
                return upstream
            try:
                rust = self._shadow(active_requests, upstream)
                if _LIVE:
                    STATE.live_steps += 1
                    if rust is None:
                        # _shadow already counted why. Executing a decision it
                        # refused to vouch for is how a scheduler starves or
                        # double-schedules a request.
                        STATE.live_fallbacks += 1
                    else:
                        return rust
            except Exception as exc:
                STATE.errors += 1
                if not STATE.reason_first_error:
                    STATE.reason_first_error = f"{type(exc).__name__}: {exc}"
                if STATE.errors >= _MAX_ERRORS:
                    STATE.disabled = True
                    print(
                        f"[trtllm-rs-shadow] disabled after {STATE.errors} errors; "
                        f"first was {STATE.reason_first_error}",
                        file=sys.stderr,
                        flush=True,
                    )
            return upstream

        def _shadow(self, active_requests: Any, upstream: Any) -> None:
            now_ms = time.perf_counter() * 1000.0

            # Iteration time, kept for comparison only. It used to be what the
            # controller was fed, and job 314929 showed why that is wrong: the
            # proxy read 11.71 ms while AIPerf measured 39.10 ms on the same
            # run, so the controller believed it was inside a 20 ms budget it
            # was in fact overshooting by 2x, and opened the cap to maximum.
            #
            # The proxy is only equal to ITL when every generating sequence
            # advances exactly one token per scheduler iteration. With chunked
            # prefill and micro-batching it does not, and the error is not small.
            if STATE.last_call_ms is not None:
                gap = now_ms - STATE.last_call_ms
                STATE.iteration_ms_ewma = (
                    gap if STATE.iteration_ms_ewma == 0.0
                    else 0.9 * STATE.iteration_ms_ewma + 0.1 * gap
                )
            STATE.last_call_ms = now_ms

            requests = list(active_requests)
            n = len(requests)
            ids: list[int] = []
            is_context: list[bool] = []
            prompt_lens: list[int] = []
            context_done: list[int] = []
            generated: list[int] = []
            max_new: list[int] = []
            arrival: list[float] = []

            for req in requests:
                rid = int(getattr(req, "py_request_id", getattr(req, "request_id", 0)))
                plen = int(getattr(req, "py_prompt_len", getattr(req, "prompt_len", 0)))
                ids.append(rid)
                is_context.append(bool(getattr(req, "is_context_init_state", False)))
                prompt_lens.append(plen)
                context_done.append(int(getattr(req, "context_current_position", 0)))
                generated.append(max(0, int(getattr(req, "max_beam_num_tokens", plen)) - plen))
                max_new.append(int(getattr(req, "py_max_new_tokens", getattr(req, "max_new_tokens", 0))))
                arrival.append(STATE.first_seen_ms.setdefault(rid, now_ms))

            # True inter-token latency, measured per sequence: how long a
            # generating request waited per token it actually produced. This is
            # the quantity the SLO is written in and the quantity AIPerf reports,
            # so it is the only honest thing to steer on.
            itl_samples: list[float] = []
            generating = 0
            live_ids: set[int] = set()
            for i, rid in enumerate(ids):
                live_ids.add(rid)
                if is_context[i]:
                    continue
                generating += 1
                STATE.generating_seen += 1
                produced = generated[i]
                previous = STATE.token_progress.get(rid)
                if previous is not None:
                    prev_ms, prev_tokens = previous
                    advanced = produced - prev_tokens
                    if advanced > 0:
                        sample = (now_ms - prev_ms) / advanced
                        itl_samples.append(sample)
                        STATE.advanced_seen += 1
                        STATE.reservoir(STATE.advancing_itl, STATE.advancing_itl_count, sample)
                        STATE.token_progress[rid] = (now_ms, produced)
                else:
                    STATE.token_progress[rid] = (now_ms, produced)
            for gone in [r for r in STATE.token_progress if r not in live_ids]:
                STATE.token_progress.pop(gone, None)

            # What the controller steers on. NOT the mean of the samples above.
            #
            # Those samples exist only for sequences that advanced this step, so
            # they describe the population that is being served and say nothing
            # about the one that is waiting. Job 316849 is the whole argument:
            # the engine held 26 generating sequences advancing every 15.4 ms
            # while AIPerf had 74 requests in decode and measured 91 ms, because
            # the other 48 had produced a first token and were then not
            # advancing at all. The advanced-only mean read 15.4 ms, the
            # controller saw itself comfortably inside a 20 ms budget, never
            # throttled, and goodput was 0.00.
            #
            # AIPerf's per-request ITL is (last_token - first_token)/(tokens-1).
            # Projecting it to now -- (now - first_token)/tokens_since_first --
            # is the same quantity with any in-progress stall included, so a
            # request that is currently stuck raises the signal while it is
            # stuck rather than only once it finishes. Steering on the p90 of
            # that across live started requests is the same shape as the pass
            # criterion, which is good_frac >= 0.90.
            # Time since each running request's last token. Equal to the step
            # time while everything advances every step; growing exactly when
            # the engine starves someone. An average anchored at the first
            # token was tried first and the simulator rejected it -- that
            # anchor precedes decode admission, so every freshly handed-off
            # request arrives carrying a latency decode concurrency cannot fix.
            projected: list[float] = []
            for i, rid in enumerate(ids):
                if is_context[i]:
                    continue
                last = STATE.token_progress.get(rid)
                if last is not None:
                    projected.append(now_ms - last[0])

            if projected:
                steer = _pct(projected, 0.90) or 0.0
                for v in projected:
                    STATE.reservoir(STATE.steer_itl, STATE.steer_itl_count, v)
                STATE.steer_itl_ms_ewma = (
                    steer if STATE.steer_itl_ms_ewma == 0.0
                    else 0.9 * STATE.steer_itl_ms_ewma + 0.1 * steer
                )
                self._rust.observe(steer, len(projected))
                STATE.observe_calls += 1

            if itl_samples:
                # Kept for reporting only: the inter-token interval of sequences
                # the engine is actually advancing. Useful for telling "the
                # engine is slow" apart from "the engine is starving requests",
                # which are different failures with different fixes.
                measured = sum(itl_samples) / len(itl_samples)
                STATE.advancing_itl_ms_ewma = (
                    measured if STATE.advancing_itl_ms_ewma == 0.0
                    else 0.9 * STATE.advancing_itl_ms_ewma + 0.1 * measured
                )

            fitting_idx, paused_idx = self._rust.decide(
                ids, is_context, prompt_lens, context_done, generated, max_new,
                arrival, [], self._free_blocks(), now_ms,
            )

            STATE.steps += 1
            if _DUMP_EVERY > 0 and STATE.steps % _DUMP_EVERY == 0:
                _dump()
            STATE.candidates_seen += n
            STATE.rust_admitted += len(fitting_idx)
            upstream_fitting = list(upstream[0]) if upstream and upstream[0] else []
            STATE.upstream_admitted += len(upstream_fitting)

            fit_set, pause_set = set(fitting_idx), set(paused_idx)
            if (any(i >= n for i in fit_set | pause_set)
                    or fit_set & pause_set):
                STATE.malformed += 1
                return None
            if n > 0 and not fit_set:
                STATE.empty_when_work_available += 1

            upstream_ids = {
                int(getattr(r, "py_request_id", getattr(r, "request_id", -1)))
                for r in upstream_fitting
            }
            if {ids[i] for i in fit_set} != upstream_ids:
                STATE.diverged += 1

            # The upstream contract is (fitting, fitting_disagg_gen_init, paused).
            # The disagg-gen-init list is passed through untouched: those
            # requests are mid-KV-handoff and admitting or pausing them is the
            # transceiver's business, not an admission policy's.
            return (
                [requests[i] for i in fitting_idx],
                list(upstream[1]) if upstream and len(upstream) > 1 else [],
                [requests[i] for i in paused_idx],
            )

        # Drop the per-request arrival clock for requests that are gone, so a
        # long run does not accumulate one float per request forever.
        def prune(self, live_ids: set[int]) -> None:
            for rid in list(STATE.first_seen_ms):
                if rid not in live_ids:
                    STATE.first_seen_ms.pop(rid, None)

    _util.BindCapacityScheduler = ShadowCapacityScheduler
    _INSTALLED = ShadowCapacityScheduler
    atexit.register(_dump)
    print("[trtllm-rs-shadow] installed", file=sys.stderr, flush=True)
    return True
