"""
RustBackedCapacityScheduler: implements the ``CapacityScheduler`` ABC
(third_party/TensorRT-LLM/tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py:296-306)
by wrapping an upstream ``CapacityScheduler`` and a ``trtllm_rs.RustScheduler``.

Hook point: this wraps ``SimpleScheduler.capacity_scheduler``, ONE layer
below the whole ``RequestScheduler``. Admission policy -- what Rust takes
over -- lives entirely in the capacity scheduler
(``fitting, fitting_disagg_gen_init, paused = capacity_scheduler.schedule_request(active_requests)``,
scheduler.py:416-418). The micro-batch scheduler one layer up
(``encoder, context, generation = micro_batch_scheduler.schedule(fitting,
inflight_request_ids)``, scheduler.py:419-421) keeps owning chunked-prefill
chunk sizing (``req.context_chunk_size``, scheduler.py:604/682), the token
budget, encoder requests, and the context/generation split -- ALL untouched.
Swapping the whole ``RequestScheduler`` would have made this bridge
responsible for that too; silently failing to set ``context_chunk_size``
would stop context requests advancing at all -- a failure that looks like a
hang, not a scheduling bug. Staying at this layer avoids that entirely.

No policy lives here beyond what Rust decides. This module never re-decides
which requests run, never "corrects" a Rust decision it thinks looks wrong,
and never falls back to picking requests itself. It flattens, calls Rust
once, maps back, and -- in shadow mode -- compares and counts. That is all.
"""

try:
    # Real import when running inside a TensorRT-LLM checkout/container.
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import CapacityScheduler
except ImportError:  # pragma: no cover - exercised only outside TensorRT-LLM
    # Falling back to `object` keeps this module importable (and therefore
    # collectible/testable) without TensorRT-LLM installed. NOTE: this makes
    # `isinstance(x, CapacityScheduler)` checks elsewhere in this package
    # vacuously true in that fallback state. That is acceptable because the
    # bridge is inert (never actually installed into a live PyExecutor)
    # without TensorRT-LLM present at runtime anyway; the check is only
    # load-bearing when TensorRT-LLM -- and the real ABC -- are present.
    CapacityScheduler = object

from ._abi import flatten, unflatten

# Cap on how many divergence / order-difference samples we keep in memory.
# Bounded so a long-running shadow session can't leak memory; counts in
# `summary()` are exact regardless of this cap.
_MAX_SAMPLES_KEPT = 50


class Divergence:
    """
    One recorded shadow-mode comparison result, kept for debugging.

    `order_differed` is only meaningful when membership agreed (this record
    lives in `order_differences()`, not `divergences()`); for a genuine
    membership divergence it is left False and should be ignored.
    """

    __slots__ = (
        "iteration",
        "upstream_fitting_ids",
        "upstream_paused_ids",
        "rust_fitting_ids",
        "rust_paused_ids",
        "order_differed",
        "upstream_num_fitting",
        "rust_num_fitting",
    )

    def __init__(self, iteration, upstream_fitting_ids, upstream_paused_ids,
                 rust_fitting_ids, rust_paused_ids, order_differed):
        self.iteration = iteration
        self.upstream_fitting_ids = upstream_fitting_ids
        self.upstream_paused_ids = upstream_paused_ids
        self.rust_fitting_ids = rust_fitting_ids
        self.rust_paused_ids = rust_paused_ids
        self.order_differed = order_differed
        # Shape of the difference, not just "they differed" -- e.g. "Rust
        # admitted 41, upstream admitted 53".
        self.upstream_num_fitting = len(upstream_fitting_ids)
        self.rust_num_fitting = len(rust_fitting_ids)

    def __repr__(self):
        return (
            f"Divergence(iteration={self.iteration}, "
            f"upstream_fitting={self.upstream_num_fitting}, "
            f"rust_fitting={self.rust_num_fitting}, "
            f"order_differed={self.order_differed})"
        )


class RustBackedCapacityScheduler(CapacityScheduler):
    """
    mode="shadow" (default): call BOTH the upstream capacity scheduler and
    Rust every step, compare their decisions, and return the UPSTREAM result
    -- the GPU always executes the upstream decision in this mode, so
    shadowing is free correctness/behavior data collected alongside any real
    job.

    Upstream (GuaranteedNoEvict / MaxUtilization) and Rust (an ITL-budget
    admission controller) are DIFFERENT POLICIES ON PURPOSE: this project
    exists because upstream was measured leaving 14% of the ITL budget
    unspent (see crates/sched/src/decode.rs on the Rust side). So in shadow
    mode a HIGH divergence rate is the EXPECTED, desired signal, not a
    failure -- and `divergence_rate == 0.0` over a nontrivial number of steps
    most likely means the patch silently did not take effect, not that
    everything is fine. See `summary()`.

    `malformed` and `empty_when_work_available` ARE failure signals, counted
    separately from ordinary divergence -- see `summary()`.

    mode="live": return the Rust decision directly for `fitting`/`paused`;
    the GPU executes it. A structurally invalid Rust decision raises instead
    of running.

    NOTE: `SimpleScheduler.can_schedule()` also calls
    `self.capacity_scheduler.schedule_request(requests)` as a fit-check for
    a not-yet-admitted request (scheduler.py's `can_schedule`), separate from
    the main per-step scheduling call. Both paths go through this class, so
    `summary()`'s `steps` counts every call to `schedule_request`, not only
    "real" scheduling steps of the main loop.
    """

    def __init__(self, upstream_capacity_scheduler, rust_scheduler, mode="shadow",
                 kv_free_blocks_fn=None):
        """
        `kv_free_blocks_fn`: optional zero-arg callable returning the current
        KV-cache free-block count as a single int. `schedule_request`'s
        fixed ABC signature (just `active_requests`) carries no
        resource-manager handle, so there is no other way for this class to
        learn `kv_free_blocks` for the `decide()` call. Defaults to `None`,
        in which case `kv_free_blocks` is reported as `0` to Rust (a
        documented gap, not a real "zero free blocks" signal). `patch.py`
        wires a real accessor from `py_executor.kv_cache_manager` when one is
        available.
        """
        if mode not in ("shadow", "live"):
            raise ValueError(f"mode must be 'shadow' or 'live', got {mode!r}")
        self._upstream = upstream_capacity_scheduler
        self._rust = rust_scheduler
        self._mode = mode
        self._kv_free_blocks_fn = kv_free_blocks_fn

        self._iteration = 0
        self._steps = 0
        self._agreed = 0
        self._diverged = 0
        self._malformed = 0
        self._empty_when_work_available = 0
        self._order_differed_count = 0
        self._divergences = []
        self._order_diffs = []

        # request_id -> monotonic ms the id was FIRST seen in active_requests
        # here. See `_arrival_ms_for()` for exactly what this does and does
        # not mean.
        self._first_seen_ms = {}

    # ---- CapacityScheduler interface ------------------------------------

    def schedule_request(self, active_requests):
        import time

        self._iteration += 1
        now_ms = time.monotonic() * 1000.0
        kv_free_blocks = self._kv_free_blocks_fn() if self._kv_free_blocks_fn else 0
        all_arrival_ms = self._arrival_ms_for(active_requests, now_ms)
        upstream_result = self._upstream.schedule_request(active_requests)
        upstream_fitting, upstream_fitting_disagg_gen_init, upstream_paused = upstream_result

        # There is no `inflight_request_ids` at the capacity-scheduler layer
        # (SimpleScheduler.schedule_request calls
        # `self.capacity_scheduler.schedule_request(active_requests)` with no
        # such argument -- inflight tracking happens one layer up, in
        # py_executor's `inflight_req_ids`). We do not have a reliable
        # source for it here, and inventing one (e.g. guessing from request
        # state) risks letting a request be scheduled into two micro-batches
        # at once, which is worse than reporting nothing. So this always
        # passes `[]` to Rust -- Rust cannot rely on this ABI to avoid
        # double-scheduling an already-inflight request at THIS layer.
        inflight = []

        arrival_by_request_id = dict(zip(
            (request.request_id for request in active_requests), all_arrival_ms
        ))
        arrival_ms = [arrival_by_request_id[request.request_id] for request in upstream_fitting]
        args = flatten(
            upstream_fitting, inflight, kv_free_blocks, now_ms, arrival_ms=arrival_ms
        )
        if self._mode == "shadow":
            try:
                rust_idx_pair = self._rust.decide(*args)
                problems = self._validate(upstream_fitting, rust_idx_pair)
            except Exception:
                self._steps += 1
                self._malformed += 1
                return upstream_result
        else:
            rust_idx_pair = self._rust.decide(*args)
            problems = self._validate(upstream_fitting, rust_idx_pair)

        if self._mode == "live":
            if problems:
                raise RuntimeError(
                    "RustScheduler.decide() returned a structurally invalid "
                    f"decision; refusing to run it. Problems: {problems}"
                )
            fitting, paused = unflatten(upstream_fitting, rust_idx_pair)
            return fitting, upstream_fitting_disagg_gen_init, [*upstream_paused, *paused]

        self._steps += 1

        if problems:
            self._malformed += 1
        else:
            fitting_idx, _paused_idx = rust_idx_pair
            if not fitting_idx and upstream_fitting:
                self._empty_when_work_available += 1
            self._record_comparison(
                upstream_fitting, upstream_paused, upstream_fitting, rust_idx_pair
            )

        return upstream_result

    def observe(self, step_ms, concurrency):
        """
        Pass-through to Rust. Must always be fed the REAL measured step time
        and REAL concurrency that actually ran on the GPU, regardless of
        mode. In shadow mode the GPU always executes the upstream decision,
        so Rust's own ItlController needs to see that reality -- feeding it
        an imagined history under its own (unexecuted) decision would make
        the shadow comparison meaningless after the first divergence. Not
        part of the `CapacityScheduler` ABC; whoever operates the executor
        loop must call `py_executor.scheduler.capacity_scheduler.observe(...)`
        explicitly.
        """
        self._rust.observe(step_ms, concurrency)

    @property
    def rust(self):
        """Read-only Rust observer used by the shadow report."""
        return self._rust

    # ---- reporting ------------------------------------------------------

    def divergences(self):
        """Bounded list of recorded MEMBERSHIP divergences (real disagreements)."""
        return list(self._divergences)

    def order_differences(self):
        """Bounded list of steps where membership agreed but ORDER did not."""
        return list(self._order_diffs)

    def summary(self):
        """
        `diverged` counts disagreement with upstream. That is the EXPECTED
        outcome here -- upstream and Rust are different policies on purpose
        (see class docstring). A HIGH `divergence_rate` is normal; a rate of
        exactly 0.0 across a meaningful number of `steps` is the suspicious
        result, not a clean pass.

        `malformed` and `empty_when_work_available` ARE failure signals,
        distinct from ordinary divergence, and are never folded into
        `diverged`.
        """
        rate = (self._diverged / self._steps) if self._steps else 0.0
        return {
            "steps": self._steps,
            "agreed": self._agreed,
            "diverged": self._diverged,
            "divergence_rate": rate,
            "malformed": self._malformed,
            "empty_when_work_available": self._empty_when_work_available,
            "order_differed": self._order_differed_count,
        }

    # ---- internals --------------------------------------------------------

    def _arrival_ms_for(self, active_requests, now_ms):
        """
        Returns a per-request arrival_ms list, same order as active_requests.

        SEMANTICS -- read carefully before using this for anything: this is
        the first monotonic timestamp (ms) at which a request id was
        observed in `active_requests` BY THIS SCHEDULER, i.e. the first time
        it became schedulable here. It is NOT when the request arrived at
        the server -- true arrival is earlier by the queueing delay that
        happens upstream of this scheduler (request queue, admission wait,
        etc.), and LlmRequest exposes no readable arrival-time attribute to
        recover that (see _abi.py's "KNOWN GAP"). This proxy IS monotonic
        and correctly ORDERED relative to other requests seen through this
        same scheduler instance, which is what deadline-feasible prefill
        batching (Moore-Hodgson, crates/sched/src/prefill.rs) needs for
        ordering -- but it must never be reported or interpreted as a true
        TTFT-relative arrival time; a deadline computed from it is biased
        early by an unknown, per-request queueing delay.

        Replace this with real arrival time once our own Rust frontend owns
        the HTTP entry point and therefore knows it directly, instead of
        relying on this scheduler-side proxy.

        Evicts ids that disappear (completed, errored, cancelled, ...) each
        call, so this cannot grow without bound across a long run.
        """
        current_ids = set()
        arrival_ms = []
        for req in active_requests:
            rid = req.request_id
            current_ids.add(rid)
            first_seen = self._first_seen_ms.get(rid)
            if first_seen is None:
                first_seen = now_ms
                self._first_seen_ms[rid] = first_seen
            arrival_ms.append(first_seen)

        for rid in (set(self._first_seen_ms) - current_ids):
            del self._first_seen_ms[rid]

        return arrival_ms

    def _fitting_disagg_gen_init_from_upstream(self, active_requests):
        """
        `fitting_disagg_gen_init` is decided by upstream's capacity scheduler
        (KV-capacity / transfer-readiness -- mechanism), not by Rust
        (admission policy for ordinary requests, which is what this class
        took over). Do not "complete" this by moving fitting-detection into
        Rust -- that question is about disaggregated-serving transfer
        readiness, not admission policy.

        Short-circuits the upstream call entirely when no request is in
        DISAGG_GENERATION_INIT state, so aggregated serving -- which never
        has any -- pays nothing for this. Only relevant in live mode: shadow
        mode already gets this field correctly from the full upstream call
        it makes for comparison.
        """
        if not any(req.is_disagg_generation_init_state for req in active_requests):
            return []
        _fitting, fitting_disagg_gen_init, _paused = self._upstream.schedule_request(active_requests)
        return fitting_disagg_gen_init

    def _validate(self, active_requests, idx_pair):
        """
        Structural validity check on a raw Rust decision. Never corrects
        anything it finds -- only reports. Checks:
          - every index is in range for `active_requests`
          - no index repeats within a single group
          - the two groups (fitting, paused) are disjoint

        There is no inflight-overlap check here: this layer always passes
        `inflight=[]` to Rust (see `schedule_request`), so such a check would
        be vacuous.
        """
        problems = []
        n = len(active_requests)
        fitting_idx, paused_idx = idx_pair
        groups = (("fitting", fitting_idx), ("paused", paused_idx))

        for name, idx_list in groups:
            for i in idx_list:
                if not (0 <= i < n):
                    problems.append(f"{name}: index {i} out of range for {n} active requests")
            if len(idx_list) != len(set(idx_list)):
                problems.append(f"{name}: duplicate index within the group")

        sets = [set(idx_list) for _, idx_list in groups]
        if len(set.union(*sets)) != sum(len(s) for s in sets):
            problems.append("fitting/paused indices are not disjoint")

        return problems

    def _record_comparison(self, upstream_fitting, upstream_paused, active_requests, rust_idx_pair):
        up_fitting_ids = frozenset(r.request_id for r in upstream_fitting)
        up_paused_ids = frozenset(r.request_id for r in upstream_paused)

        fitting, paused = unflatten(active_requests, rust_idx_pair)
        rust_fitting_ids = frozenset(r.request_id for r in fitting)
        rust_paused_ids = frozenset(
            r.request_id for r in [*upstream_paused, *paused]
        )

        membership_agrees = (
            up_fitting_ids == rust_fitting_ids and up_paused_ids == rust_paused_ids
        )

        if membership_agrees:
            self._agreed += 1
            order_differed = (
                [r.request_id for r in upstream_fitting] != [r.request_id for r in fitting]
            )
            if order_differed:
                self._order_differed_count += 1
                if len(self._order_diffs) < _MAX_SAMPLES_KEPT:
                    self._order_diffs.append(Divergence(
                        self._iteration, up_fitting_ids, up_paused_ids,
                        rust_fitting_ids, rust_paused_ids, order_differed=True,
                    ))
            return

        self._diverged += 1
        if len(self._divergences) < _MAX_SAMPLES_KEPT:
            self._divergences.append(Divergence(
                self._iteration, up_fitting_ids, up_paused_ids,
                rust_fitting_ids, rust_paused_ids, order_differed=False,
            ))
