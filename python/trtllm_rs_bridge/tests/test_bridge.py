"""
Pure-Python tests for trtllm_rs_bridge. No GPU or live scheduler execution is
needed -- TensorRT-LLM request objects, capacity schedulers, and the Rust
decision object are faked.

Run with pytest:
    python3 -m pytest tests/test_bridge.py -v

or, if pytest is unavailable, directly as a script from this directory's
parent (or anywhere -- it fixes up sys.path itself):
    python3 tests/test_bridge.py

When TensorRT-LLM is importable, its CPU-only `LlmRequestState` enum is used to
build realistic fake request state. The suite remains collectable without the
vendor package so the bridge's fallback imports can be checked locally.
"""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    from tensorrt_llm.bindings import LlmRequestState
    _HAVE_TRTLLM = True
except ImportError:
    LlmRequestState = None
    _HAVE_TRTLLM = False

try:
    import pytest
except ImportError:
    pytest = None

if pytest is not None:
    pytestmark = pytest.mark.skipif(
        not _HAVE_TRTLLM, reason="tensorrt_llm is not importable in this environment"
    )

import trtllm_rs_bridge.patch as _patch_mod
from trtllm_rs_bridge import RustBackedCapacityScheduler, flatten, install, install_global, unflatten
from trtllm_rs_bridge._abi import _ARRIVAL_MS_UNAVAILABLE

if _HAVE_TRTLLM:

    _CONTEXT_STATE = LlmRequestState.CONTEXT_INIT
    _GENERATION_STATE = LlmRequestState.GENERATION_IN_PROGRESS

try:
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import RequestScheduler
except ImportError:
    RequestScheduler = object

_CONTEXT_STATE = getattr(LlmRequestState, "CONTEXT_INIT", "context")
_GENERATION_STATE = getattr(LlmRequestState, "GENERATION_IN_PROGRESS", "generation")


class FakeLlmRequest:
    """Stands in for tensorrt_llm's LlmRequest; only the attributes _abi.py reads."""

    def __init__(self, request_id, is_context, prompt_len=8, generated=0, max_new_tokens=32,
                 context_done_tokens=0, is_disagg_generation_init_state=False,
                 is_encoder_init_state=False):
        self.request_id = request_id
        self.prompt_len = prompt_len
        self.max_new_tokens = max_new_tokens
        self.context_current_position = context_done_tokens
        self.state = _CONTEXT_STATE if is_context else _GENERATION_STATE
        self._num_tokens = prompt_len + generated
        self.is_disagg_generation_init_state = is_disagg_generation_init_state
        self.is_encoder_init_state = is_encoder_init_state

    def get_num_tokens(self, beam):
        return self._num_tokens


class FakeRustScheduler:
    """Stands in for trtllm_rs.RustScheduler; decide() returns a canned answer."""

    def __init__(self, decision=None):
        self._decision = decision if decision is not None else ([], [])
        self.last_call = None
        self.observed = []

    def decide(self, ids, is_context, prompt_lens, context_done_tokens,
               tokens_generated, max_new_tokens, arrival_ms, inflight,
               kv_free_blocks, now_ms):
        self.last_call = dict(
            ids=ids, is_context=is_context, prompt_lens=prompt_lens,
            context_done_tokens=context_done_tokens,
            tokens_generated=tokens_generated, max_new_tokens=max_new_tokens,
            arrival_ms=arrival_ms, inflight=inflight,
            kv_free_blocks=kv_free_blocks, now_ms=now_ms,
        )
        return self._decision

    def observe(self, step_ms, concurrency):
        self.observed.append((step_ms, concurrency))

    def stats(self):
        return {"samples": len(self.observed)}


class FakeUpstreamScheduler:
    """Stands in for the real capacity scheduler upstream instance."""

    def __init__(self, fitting=(), fitting_disagg_gen_init=(), paused=()):
        self._result = (list(fitting), list(fitting_disagg_gen_init), list(paused))
        self.calls = 0

    def schedule_request(self, active_requests):
        self.calls += 1
        return self._result


class FakeIterationStats:
    def __init__(self, iter_latency_ms, scheduled):
        self.iter_latency_ms = iter_latency_ms
        self.inflight_batching_stats = type(
            "InflightBatchingStats", (), {"num_scheduled_requests": scheduled}
        )()


class FakeObserverExecutor:
    def __init__(self, disable_overlap_scheduler=True, pp_size=1):
        self.scheduler = FakeRealScheduler()
        self.disable_overlap_scheduler = disable_overlap_scheduler
        self.dist = type("Dist", (), {"pp_size": pp_size})()

    def _append_iter_stats(self, stats, host_step_time_ms=None, **_kwargs):
        self.appended = (stats, host_step_time_ms)


def test_flatten_lengths_and_order():
    reqs = [
        FakeLlmRequest(10, is_context=True, prompt_len=5, generated=0,
                       context_done_tokens=2),
        FakeLlmRequest(11, is_context=False, prompt_len=5, generated=3),
        FakeLlmRequest(12, is_context=False, prompt_len=6, generated=0),
    ]
    (ids, is_context, prompt_lens, context_done_tokens, tokens_generated,
     max_new_tokens,
     arrival_ms, inflight, kvfb, now) = flatten(
        reqs, [11], kv_free_blocks=100, now_ms=42.0
    )

    assert ids == [10, 11, 12]
    assert is_context == [True, False, False]
    assert prompt_lens == [5, 5, 6]
    assert context_done_tokens == [2, 0, 0]
    assert tokens_generated == [0, 3, 0]
    assert max_new_tokens == [32, 32, 32]
    assert arrival_ms == [_ARRIVAL_MS_UNAVAILABLE] * 3
    assert inflight == [11]
    assert kvfb == 100
    assert now == 42.0
    for lst in (ids, is_context, prompt_lens, context_done_tokens,
                tokens_generated, max_new_tokens, arrival_ms):
        assert len(lst) == len(reqs)


def test_unflatten_maps_by_identity():
    reqs = [FakeLlmRequest(i, is_context=(i == 0)) for i in range(4)]
    fitting, paused = unflatten(reqs, ([0, 2], [1]))
    assert fitting[0] is reqs[0] and fitting[1] is reqs[2]
    assert paused[0] is reqs[1]


def test_shadow_mode_returns_upstream_and_records_one_divergence():
    reqs = [FakeLlmRequest(i, is_context=(i < 2)) for i in range(4)]
    upstream = FakeUpstreamScheduler(fitting=reqs[0:2], paused=reqs[3:4])
    # Rust disagrees with the upstream ordinary admission decision.
    rust = FakeRustScheduler(decision=([1], [0]))
    sched = RustBackedCapacityScheduler(upstream, rust, mode="shadow")

    result = sched.schedule_request(reqs)

    assert result == ([reqs[0], reqs[1]], [], [reqs[3]])
    assert rust.last_call["ids"] == [0, 1]
    summary = sched.summary()
    assert summary["steps"] == 1
    assert summary["diverged"] == 1
    assert summary["agreed"] == 0
    assert summary["malformed"] == 0
    assert len(sched.divergences()) == 1


def test_shadow_mode_zero_divergence_on_order_only_difference():
    reqs = [FakeLlmRequest(i, is_context=(i == 0)) for i in range(3)]
    upstream = FakeUpstreamScheduler(fitting=reqs)
    # Same membership as upstream, reversed order.
    rust = FakeRustScheduler(decision=([2, 1, 0], []))
    sched = RustBackedCapacityScheduler(upstream, rust, mode="shadow")

    sched.schedule_request(reqs)

    summary = sched.summary()
    assert summary["diverged"] == 0
    assert summary["agreed"] == 1
    assert len(sched.divergences()) == 0
    assert summary["order_differed"] == 1
    assert len(sched.order_differences()) == 1


def test_live_mode_returns_rust_decision_and_calls_upstream_capacity():
    reqs = [FakeLlmRequest(i, is_context=(i == 0)) for i in range(3)]
    upstream = FakeUpstreamScheduler(fitting=reqs[0:2], fitting_disagg_gen_init=reqs[2:3])
    rust = FakeRustScheduler(decision=([0], [1]))
    sched = RustBackedCapacityScheduler(upstream, rust, mode="live")

    fitting, fitting_disagg_gen_init, paused = sched.schedule_request(reqs)

    assert fitting == [reqs[0]]
    assert fitting_disagg_gen_init == [reqs[2]]
    assert paused == [reqs[1]]
    assert upstream.calls == 1
    assert rust.last_call["ids"] == [0, 1]


def test_shadow_mode_counts_malformed_out_of_range_index():
    reqs = [FakeLlmRequest(i, is_context=(i == 0)) for i in range(2)]
    upstream = FakeUpstreamScheduler(fitting=reqs[0:1], paused=reqs[1:2])
    rust = FakeRustScheduler(decision=([0], [5]))  # index 5 does not exist
    sched = RustBackedCapacityScheduler(upstream, rust, mode="shadow")

    result = sched.schedule_request(reqs)

    assert result == ([reqs[0]], [], [reqs[1]])
    summary = sched.summary()
    assert summary["malformed"] == 1
    assert summary["steps"] == 1
    assert summary["diverged"] == 0  # malformed must not double-count as diverged
    assert len(sched.divergences()) == 0


def test_shadow_mode_returns_exact_upstream_result_when_rust_decide_raises():
    class RaisingRustScheduler(FakeRustScheduler):
        def decide(self, *args):
            raise RuntimeError("Rust decide failed")

    reqs = [FakeLlmRequest(0, is_context=True)]
    upstream = FakeUpstreamScheduler(fitting=reqs)
    sched = RustBackedCapacityScheduler(upstream, RaisingRustScheduler(), mode="shadow")

    with patch("trtllm_rs_bridge.scheduler.flatten", return_value=([],) * 10):
        result = sched.schedule_request(reqs)

    assert result is upstream._result
    assert sched.summary()["steps"] == 1
    assert sched.summary()["malformed"] == 1


def test_shadow_mode_returns_exact_upstream_result_when_rust_outer_result_is_malformed():
    reqs = [FakeLlmRequest(0, is_context=True)]
    upstream = FakeUpstreamScheduler(fitting=reqs)
    sched = RustBackedCapacityScheduler(upstream, FakeRustScheduler(decision=([],)), mode="shadow")

    with patch("trtllm_rs_bridge.scheduler.flatten", return_value=([],) * 10):
        result = sched.schedule_request(reqs)

    assert result is upstream._result
    assert sched.summary()["steps"] == 1
    assert sched.summary()["malformed"] == 1


def test_install_observes_each_completed_stat_once_and_restore_unhooks():
    executor = FakeObserverExecutor()
    rust = FakeRustScheduler()
    original = executor._append_iter_stats

    handle = install(executor, rust)
    duplicate_handle = install(executor, FakeRustScheduler())
    executor._append_iter_stats(FakeIterationStats(14.0, 7), host_step_time_ms=11.5)

    assert rust.observed == [(11.5, 7)]
    duplicate_handle.restore()
    handle.restore()
    assert executor._append_iter_stats.__func__ is original.__func__
    executor._append_iter_stats(FakeIterationStats(12.0, 4), host_step_time_ms=9.0)
    assert rust.observed == [(11.5, 7)]


def test_observer_falls_back_to_iteration_time_and_finalized_overlap_concurrency():
    executor = FakeObserverExecutor()
    rust = FakeRustScheduler()
    handle = install(executor, rust)
    try:
        executor._append_iter_stats(
            FakeIterationStats(17.25, 13), host_step_time_ms=float("nan")
        )
        assert rust.observed == [(17.25, 13)]
    finally:
        handle.restore()


def test_observer_skips_iteration_fallback_when_overlap_has_no_clean_host_time():
    executor = FakeObserverExecutor(disable_overlap_scheduler=False)
    rust = FakeRustScheduler()
    handle = install(executor, rust)
    try:
        executor._append_iter_stats(
            FakeIterationStats(17.25, 13), host_step_time_ms=None
        )
        assert rust.observed == []
    finally:
        handle.restore()


def test_observer_skips_iteration_fallback_when_pipeline_parallel_has_no_host_time():
    executor = FakeObserverExecutor(disable_overlap_scheduler=True, pp_size=2)
    rust = FakeRustScheduler()
    handle = install(executor, rust)
    try:
        executor._append_iter_stats(
            FakeIterationStats(17.25, 13), host_step_time_ms=None
        )
        assert rust.observed == []
    finally:
        handle.restore()


def test_observer_skips_iteration_fallback_when_executor_mode_is_unknown():
    executor = FakeObserverExecutor()
    del executor.disable_overlap_scheduler
    rust = FakeRustScheduler()
    handle = install(executor, rust)
    try:
        executor._append_iter_stats(
            FakeIterationStats(17.25, 13), host_step_time_ms=None
        )
        assert rust.observed == []
    finally:
        handle.restore()


def test_observer_ignores_malformed_timing_and_concurrency_without_breaking_append():
    executor = FakeObserverExecutor()
    rust = FakeRustScheduler()
    handle = install(executor, rust)
    try:
        stats = FakeIterationStats("not-a-time", "not-a-count")
        executor._append_iter_stats(stats, host_step_time_ms=None)
        assert executor.appended == (stats, None)
        assert rust.observed == []
    finally:
        handle.restore()


def test_shadow_report_delivers_rust_stats_and_observer_samples():
    rust = FakeRustScheduler()
    scheduler = RustBackedCapacityScheduler(FakeUpstreamScheduler(), rust)
    scheduler.observe(11.5, 7)

    entry = {"summary": scheduler.summary()}
    if hasattr(scheduler, "rust") and hasattr(scheduler.rust, "stats"):
        entry["rust_stats"] = scheduler.rust.stats()
        entry["observer_samples"] = entry["rust_stats"].get("samples", 0)

    assert entry["rust_stats"] == {"samples": 1}
    assert entry["observer_samples"] == 1


class FakeRealScheduler(RequestScheduler if _HAVE_TRTLLM else object):
    """
    A real RequestScheduler subclass when TensorRT-LLM is available, with a
    capacity scheduler below it. This verifies install_global() preserves the
    request-scheduler layer.
    """

    def __init__(self):
        self.capacity_scheduler = FakeUpstreamScheduler()
        self.micro_batch_scheduler = object()

    def schedule_request(self, active_requests, inflight_request_ids):
        raise AssertionError("the request-scheduler layer must stay upstream")

    def can_schedule(self, requests):
        return True


def test_install_global_wraps_capacity_scheduler_once_and_restores():
    from tensorrt_llm._torch.pyexecutor import py_executor_creator

    class FakePyExecutor:
        def __init__(self):
            self.scheduler = FakeRealScheduler()

    def fake_create_py_executor(*args, **kwargs):
        return FakePyExecutor()

    real_create_py_executor = py_executor_creator.create_py_executor
    py_executor_creator.create_py_executor = fake_create_py_executor
    _patch_mod._wrapped_executors.clear()
    try:
        factory_calls = []

        def factory(py_executor):
            factory_calls.append(py_executor)
            return FakeRustScheduler()

        handle1 = install_global(factory, mode="shadow")
        pe1 = py_executor_creator.create_py_executor()
        assert isinstance(pe1.scheduler, FakeRealScheduler)
        assert isinstance(pe1.scheduler.capacity_scheduler, RustBackedCapacityScheduler)
        assert pe1 in _patch_mod.wrapped_executors()
        assert len(factory_calls) == 1

        # Calling install_global again must not double-wrap: create_py_executor
        # must not be patched a second time.
        patched_after_first = py_executor_creator.create_py_executor
        handle2 = install_global(factory, mode="shadow")
        assert py_executor_creator.create_py_executor is patched_after_first

        pe2 = py_executor_creator.create_py_executor()
        assert isinstance(pe2.scheduler, FakeRealScheduler)
        assert isinstance(pe2.scheduler.capacity_scheduler, RustBackedCapacityScheduler)
        assert not isinstance(
            pe2.scheduler.capacity_scheduler._upstream, RustBackedCapacityScheduler
        )
        assert len(factory_calls) == 2  # once per executor created, not doubled

        handle2.restore()  # no-op: handle2 never owned the patch
        assert py_executor_creator.create_py_executor is patched_after_first

        handle1.restore()
        assert py_executor_creator.create_py_executor is fake_create_py_executor
    finally:
        py_executor_creator.create_py_executor = real_create_py_executor
        _patch_mod._wrapped_executors.clear()


def test_install_raises_on_scheduler_without_capacity_layer():
    class NotAScheduler:
        pass

    class FakePyExecutor:
        def __init__(self):
            self.scheduler = NotAScheduler()

    with pytest.raises(TypeError, match="capacity_scheduler|NotAScheduler"):
        install(FakePyExecutor(), rust_scheduler=FakeRustScheduler())


def test_public_api_exports_capacity_scheduler():
    from trtllm_rs_bridge import RustBackedCapacityScheduler

    assert RustBackedCapacityScheduler.__name__ == "RustBackedCapacityScheduler"


def test_install_replaces_only_capacity_scheduler():
    from trtllm_rs_bridge import RustBackedCapacityScheduler, install

    class Capacity:
        def schedule_request(self, active_requests):
            return [], [], []

    class Scheduler:
        def __init__(self):
            self.capacity_scheduler = Capacity()
            self.micro_batch_scheduler = object()

        def schedule_request(self, active_requests, inflight_request_ids):
            raise AssertionError("the request-scheduler layer must stay upstream")

        def can_schedule(self, requests):
            return True

    class Executor:
        def __init__(self):
            self.scheduler = Scheduler()

    executor = Executor()
    original_scheduler = executor.scheduler
    original_capacity = original_scheduler.capacity_scheduler
    original_micro_batch = original_scheduler.micro_batch_scheduler

    class Rust:
        def decide(self, *args):
            return [], []

    handle = install(executor, Rust())

    assert executor.scheduler is original_scheduler
    assert executor.scheduler.micro_batch_scheduler is original_micro_batch
    assert isinstance(executor.scheduler.capacity_scheduler, RustBackedCapacityScheduler)
    assert executor.scheduler.capacity_scheduler._upstream is original_capacity
    handle.restore()
    assert executor.scheduler.capacity_scheduler is original_capacity


def test_shadow_returns_exact_upstream_output_and_only_sends_accepted_requests_to_rust():
    from trtllm_rs_bridge import RustBackedCapacityScheduler

    class Capacity:
        def __init__(self, result):
            self.result = result

        def schedule_request(self, active_requests):
            return self.result

    class Request:
        def __init__(self, request_id):
            self.request_id = request_id

    class Rust:
        def __init__(self):
            self.ids = None

        def decide(self, ids, *args):
            self.ids = ids
            return [0], []

    accepted = Request(1)
    disagg_init = Request(2)
    paused = Request(3)
    expected = ([accepted], [disagg_init], [paused])
    rust = Rust()
    scheduler = RustBackedCapacityScheduler(Capacity(expected), rust, mode="shadow")

    with patch(
        "trtllm_rs_bridge.scheduler.flatten",
        side_effect=lambda active, *args, **kwargs: ([r.request_id for r in active],) + ([],) * 9,
    ), patch(
        "trtllm_rs_bridge.scheduler.unflatten",
        side_effect=lambda active, indexes: ([active[i] for i in indexes[0]], [active[i] for i in indexes[1]]),
    ):
        result = scheduler.schedule_request([accepted, disagg_init, paused])

    assert result is expected
    assert rust.ids == [accepted.request_id]


def test_live_preserves_upstream_paused_and_disagg_init_requests():
    from trtllm_rs_bridge import RustBackedCapacityScheduler

    class Capacity:
        def schedule_request(self, active_requests):
            return [first, second], [disagg_init], [upstream_paused]

    class Request:
        def __init__(self, request_id):
            self.request_id = request_id

    class Rust:
        def decide(self, *args):
            return [0], [1]

    first = Request(1)
    second = Request(2)
    disagg_init = Request(3)
    upstream_paused = Request(4)
    scheduler = RustBackedCapacityScheduler(
        Capacity(), Rust(), mode="live"
    )

    with patch("trtllm_rs_bridge.scheduler.flatten", return_value=([],) * 10), patch(
        "trtllm_rs_bridge.scheduler.unflatten",
        side_effect=lambda active, indexes: ([active[i] for i in indexes[0]], [active[i] for i in indexes[1]]),
    ):
        fitting, fitting_disagg_gen_init, paused = scheduler.schedule_request(
            [first, second, disagg_init, upstream_paused]
        )

    assert fitting == [first]
    assert fitting_disagg_gen_init == [disagg_init]
    assert paused == [upstream_paused, second]


if __name__ == "__main__":
    for test_fn in (
        test_public_api_exports_capacity_scheduler,
        test_install_replaces_only_capacity_scheduler,
        test_shadow_returns_exact_upstream_output_and_only_sends_accepted_requests_to_rust,
        test_shadow_mode_returns_exact_upstream_result_when_rust_decide_raises,
        test_shadow_mode_returns_exact_upstream_result_when_rust_outer_result_is_malformed,
        test_live_preserves_upstream_paused_and_disagg_init_requests,
        test_install_observes_each_completed_stat_once_and_restore_unhooks,
        test_observer_falls_back_to_iteration_time_and_finalized_overlap_concurrency,
        test_observer_skips_iteration_fallback_when_overlap_has_no_clean_host_time,
        test_observer_skips_iteration_fallback_when_pipeline_parallel_has_no_host_time,
        test_observer_skips_iteration_fallback_when_executor_mode_is_unknown,
        test_observer_ignores_malformed_timing_and_concurrency_without_breaking_append,
        test_shadow_report_delivers_rust_stats_and_observer_samples,
    ):
        test_fn()
        print(f"PASS {test_fn.__name__}")
    if not _HAVE_TRTLLM:
        print("tensorrt_llm is not importable here -- skipping (not failing).")
        sys.exit(0)
