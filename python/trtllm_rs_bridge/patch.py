"""
Installation: swap `PyExecutor.scheduler.capacity_scheduler` for a
`RustBackedCapacityScheduler`.

Two entry points, for two different ways of getting hold of a PyExecutor:

  install(py_executor, rust_scheduler, mode="shadow") -> handle
      Use this when the caller already holds a live PyExecutor instance.
      Swaps `py_executor.scheduler` in place; `handle.restore()` puts the
      original back.

  install_global(rust_scheduler_factory, mode="shadow") -> handle
      `trtllm-serve` builds its own PyExecutor deep inside
      `tensorrt_llm._torch.pyexecutor.py_executor_creator.create_py_executor(...)`
      and never hands that instance to any caller -- there is no point at
      which `install()` could be invoked on that path. `install_global()`
      instead wraps `create_py_executor` itself, so every PyExecutor it
      returns already has its scheduler swapped. `rust_scheduler_factory` is
      called with the just-created PyExecutor (so it can read sizing
      arguments -- max_batch_size, max_num_tokens, kv_total_blocks -- off the
      executor, since those only exist once the executor exists) and must
      return a `trtllm_rs.RustScheduler`.

`install_global()` is idempotent: calling it twice does not double-wrap.
Double-wrapping would make the shadow comparison compare Rust against Rust,
which always reports zero divergences no matter what the real policies do --
the most dangerous kind of silent failure for a shadow gate.
"""

try:
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import RequestScheduler
except ImportError:  # pragma: no cover - exercised only outside TensorRT-LLM
    RequestScheduler = object  # see scheduler.py for why this is an accepted tradeoff

import math

from .scheduler import RustBackedCapacityScheduler

_wrapped_executors = []


class InstallHandle:
    """Returned by install()/install_global(). restore() is idempotent."""

    def __init__(self, restore_fn):
        self._restore_fn = restore_fn
        self._restored = False

    def restore(self):
        if not self._restored:
            self._restore_fn()
            self._restored = True


def _kv_free_blocks_fn_for(py_executor):
    """
    Best-effort accessor for the current KV-cache free-block count, wired
    from `py_executor.kv_cache_manager` when one is present (confirmed
    attribute: py_executor.py:415,639 set
    `self.kv_cache_manager = resource_manager.resource_managers.get(...)`).

    `kv_cache_manager.get_kv_cache_stats().num_free_blocks_per_window_size`
    is a dict keyed by attention window size (confirmed usage:
    scheduler.py's NoEvictScheduledBlocksManager.__init__). The Rust ABI
    given to this bridge takes a single scalar `kv_free_blocks: int`, so
    multiple window sizes are collapsed by summation here. This is a real
    simplification when sliding-window / variable-window attention is in
    play (multiple pools get flattened into one number) -- flagged in this
    task's report, not silently assumed correct.

    Returns None if no kv_cache_manager is present (e.g. in tests), in which
    case RustBackedCapacityScheduler reports kv_free_blocks=0 to Rust.
    """
    kv_cache_manager = getattr(py_executor, "kv_cache_manager", None)
    if kv_cache_manager is None:
        return None

    def _fn():
        stats = kv_cache_manager.get_kv_cache_stats()
        return sum(stats.num_free_blocks_per_window_size.values())

    return _fn


def _clean_positive_float(value):
    """Return a finite positive millisecond value, or None for bad telemetry."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0.0 else None


def _known_non_overlap_executor(py_executor):
    """Return True only when the executor proves a clean non-overlap mode."""
    if getattr(py_executor, "disable_overlap_scheduler", None) is not True:
        return False
    dist = getattr(py_executor, "dist", None)
    try:
        pp_size = int(getattr(dist, "pp_size", None))
    except (TypeError, ValueError):
        return False
    return pp_size == 1


def _observe_completed_record(
    rust_scheduler, stats, host_step_time_ms, *, allow_iteration_fallback
):
    """Deliver one finalized iteration record without perturbing its owner."""
    step_ms = _clean_positive_float(host_step_time_ms)
    if step_ms is None and allow_iteration_fallback:
        step_ms = _clean_positive_float(getattr(stats, "iter_latency_ms", None))

    inflight = getattr(stats, "inflight_batching_stats", None)
    concurrency = getattr(inflight, "num_scheduled_requests", None)
    if isinstance(concurrency, bool):
        return
    try:
        concurrency = int(concurrency)
    except (TypeError, ValueError):
        return
    if step_ms is None or concurrency < 0:
        return

    try:
        rust_scheduler.observe(step_ms, concurrency)
    except Exception:
        # Observation is telemetry; the vendor executor remains authoritative.
        return


def _install_observer(py_executor, rust_scheduler):
    """Wrap this executor's completed-record append path exactly once."""
    original = getattr(py_executor, "_append_iter_stats", None)
    if not callable(original):
        return lambda: None
    allow_iteration_fallback = _known_non_overlap_executor(py_executor)

    def _append_iter_stats(stats, *args, **kwargs):
        result = original(stats, *args, **kwargs)
        host_step_time_ms = kwargs.get("host_step_time_ms")
        if host_step_time_ms is None and len(args) >= 4:
            host_step_time_ms = args[3]
        _observe_completed_record(
            rust_scheduler,
            stats,
            host_step_time_ms,
            allow_iteration_fallback=allow_iteration_fallback,
        )
        return result

    py_executor._append_iter_stats = _append_iter_stats

    def _restore():
        if py_executor._append_iter_stats is _append_iter_stats:
            py_executor._append_iter_stats = original

    return _restore


def install(py_executor, rust_scheduler, mode="shadow"):
    """
    Swap only `py_executor.scheduler.capacity_scheduler` for a
    `RustBackedCapacityScheduler`. The RequestScheduler and its micro-batch
    scheduler remain upstream-owned.
    """
    existing = getattr(py_executor, "_trtllm_rs_bridge_install", None)
    if existing is not None:
        return InstallHandle(lambda: None)

    original = py_executor.scheduler
    if not isinstance(original, RequestScheduler):
        raise TypeError(
            "install() expected py_executor.scheduler to already satisfy the "
            "RequestScheduler interface, but it is "
            f"{type(original).__module__}.{type(original).__qualname__}; "
            "refusing to install to avoid a silent no-op."
        )

    original_capacity_scheduler = getattr(original, "capacity_scheduler", None)
    if original_capacity_scheduler is None:
        raise TypeError(
            "install() expected py_executor.scheduler to expose capacity_scheduler; "
            f"got {type(original).__module__}.{type(original).__qualname__}."
        )

    wrapped = RustBackedCapacityScheduler(
        original_capacity_scheduler, rust_scheduler, mode=mode,
        kv_free_blocks_fn=_kv_free_blocks_fn_for(py_executor),
    )
    original.capacity_scheduler = wrapped
    restore_observer = _install_observer(py_executor, rust_scheduler)
    py_executor._trtllm_rs_bridge_install = wrapped

    def _restore():
        if original.capacity_scheduler is wrapped:
            original.capacity_scheduler = original_capacity_scheduler
        restore_observer()
        if getattr(py_executor, "_trtllm_rs_bridge_install", None) is wrapped:
            del py_executor._trtllm_rs_bridge_install

    return InstallHandle(_restore)


def wrapped_executors():
    """Live PyExecutor instances that install_global() has swapped the scheduler on."""
    return list(_wrapped_executors)


def install_global(rust_scheduler_factory, mode="shadow"):
    """
    Monkeypatch
    `tensorrt_llm._torch.pyexecutor.py_executor_creator.create_py_executor`
    (confirmed signature:
    `create_py_executor(llm_args, checkpoint_dir=None, tokenizer=None,
    profiling_stage_data=None, resource_governor_queue=None) -> PyExecutor`,
    py_executor_creator.py:336) so every PyExecutor it builds comes back with
    `.scheduler` already swapped.

    Idempotent: if `create_py_executor` is already our patched function
    (checked via a marker attribute, not a separate global flag, so a second
    call is safe even if state was otherwise reset), this returns a handle
    whose `restore()` is a no-op instead of wrapping a second time.
    """
    from tensorrt_llm._torch.pyexecutor import py_executor_creator

    current = py_executor_creator.create_py_executor
    if getattr(current, "_is_trtllm_rs_bridge_patch", False):
        return InstallHandle(lambda: None)

    original_create = current

    def _patched_create_py_executor(*args, **kwargs):
        py_executor = original_create(*args, **kwargs)
        rust_scheduler = rust_scheduler_factory(py_executor)
        install(py_executor, rust_scheduler, mode=mode)
        _wrapped_executors.append(py_executor)
        return py_executor

    _patched_create_py_executor._is_trtllm_rs_bridge_patch = True
    _patched_create_py_executor.__wrapped__ = original_create
    py_executor_creator.create_py_executor = _patched_create_py_executor

    def _restore():
        if py_executor_creator.create_py_executor is _patched_create_py_executor:
            py_executor_creator.create_py_executor = original_create

    return InstallHandle(_restore)
