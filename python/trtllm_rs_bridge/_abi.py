"""
Python <-> Rust crossing for the CAPACITY-scheduler swap.

Measured cost: one PyO3 call crossing is 9,533 ns. With up to ~60 active
requests per step, reading attributes per request individually through Rust
would mean hundreds of crossings, i.e. multiple ms per step -- larger than
the entire step budget. So this module:

  1. flattens ``active_requests`` into parallel Python lists with plain list
     comprehensions (pure Python iteration, zero FFI calls),
  2. crosses into Rust exactly ONCE via ``RustScheduler.decide(...)``,
  3. maps the returned index pair back onto ``LlmRequest`` objects with one
     more comprehension.

No per-request FFI call is ever made from this module.

Hook point: this bridge swaps ``SimpleScheduler.capacity_scheduler`` (a
``CapacityScheduler``), not the whole ``RequestScheduler``. Admission policy
-- what Rust takes over -- lives entirely in the capacity scheduler; the
micro-batch scheduler one layer up keeps owning chunked-prefill chunk sizing,
the token budget, encoder requests, and the context/generation split, all
untouched. See ``scheduler.RustBackedCapacityScheduler`` for why.
``CapacityScheduler.schedule_request(active_requests) -> (fitting,
fitting_disagg_gen_init, paused)`` takes ONE argument -- there is no
``inflight_request_ids`` at this layer (confirmed:
``SimpleScheduler.schedule_request`` calls
``self.capacity_scheduler.schedule_request(active_requests)`` with no
inflight argument, third_party/TensorRT-LLM/tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py:416-418).

Attribute names below were read directly out of this checkout, not guessed,
and cross-checked live against the container's installed TensorRT-LLM 1.3.0rc22:

  request id            -> req.request_id
      nanobind: ``.def_rw("request_id", &GenLlmReq::mRequestId)``
      (third_party/TensorRT-LLM/cpp/tensorrt_llm/nanobind/batch_manager/bindings.cpp:132)

  context vs generation phase -> req.state == LlmRequestState.CONTEXT_INIT
      ``req.state`` is nanobind ``.def_prop_rw("state", ...)`` (bindings.cpp:136).
      Structural read of request lifecycle state, not a scheduling decision.

  prompt length          -> req.prompt_len
      nanobind: ``.def_rw("prompt_len", &GenLlmReq::mPromptLen)`` (bindings.cpp:133)

  context tokens already processed -> req.context_current_position
      nanobind: ``.def_prop_rw("context_current_position",
      &GenLlmReq::getContextCurrentPosition, &GenLlmReq::setContextCurrentPosition)``
      (bindings.cpp:171). With chunked prefill a context request can be
      partway through its prompt across several steps; Rust's
      ``PendingPrefill.done_tokens`` (crates/sched/src/prefill.rs) needs this
      so it does not re-charge the whole prompt every step for an
      already-partly-processed request. Only meaningful while a request is
      still in context/prefill -- for a generation-phase request this is
      whatever value it held when context finished (normally its full
      ``prompt_len``), which Rust should treat as "context already done",
      not act on for admission.

  tokens generated so far -> req.get_num_tokens(0) - req.prompt_len
      ``get_num_tokens(beam)`` is nanobind
      ``.def("get_num_tokens", &GenLlmReq::getNumTokens, nb::arg("beam"))``
      (bindings.cpp:99), total tokens (prompt + generated) for that beam.
      No dedicated "tokens generated so far" attribute exists, so it is
      computed here as a plain subtraction -- data extraction, not policy.

  max new tokens          -> req.max_new_tokens
      nanobind: ``.def_rw("max_new_tokens", &GenLlmReq::mMaxNewTokens)`` (bindings.cpp:134)

KNOWN GAP -- arrival time: no readable arrival-time attribute exists on
LlmRequest in this checkout (confirmed again directly against the
container's binding: ``dir()`` shows only state predicates, no arrival
attribute of any kind). Because ``flatten()`` is a pure, stateless mapper,
it cannot fix this on its own -- there is nothing to cache in a function
with no persistent state. So ``flatten()`` takes an optional ``arrival_ms``
override: pass a precomputed per-request list when you have one (see
``scheduler.RustBackedCapacityScheduler``, which maintains a real
first-seen-here proxy for this), and it is used verbatim; omit it and every
request gets the ``_ARRIVAL_MS_UNAVAILABLE`` sentinel.

KNOWN GAP -- inflight: at the capacity-scheduler layer there is no
``inflight_request_ids`` input at all (see "Hook point" above). This module
always reports ``inflight=[]`` to Rust rather than inventing a source for
it -- a wrong inflight set would let a request be scheduled into two
micro-batches at once, which is worse than reporting nothing. See
``scheduler.py`` for where this is decided and why.
"""

# Sentinel used for arrival_ms when no override is supplied (see "KNOWN GAP"
# above). Never treat this as a real timestamp.
_ARRIVAL_MS_UNAVAILABLE = 0.0


def flatten(active_requests, inflight, kv_free_blocks, now_ms, arrival_ms=None):
    """
    Flatten ``active_requests`` (a list of LlmRequest) into the parallel-array
    shape ``RustScheduler.decide()`` takes. One Python-level pass, zero FFI
    calls anywhere in this function.

    ``inflight`` is passed straight through uninterpreted -- see the module
    docstring's "KNOWN GAP -- inflight"; callers at the capacity-scheduler
    layer should pass ``[]``.

    ``arrival_ms``, if given, must already be a per-request list the same
    length as ``active_requests``, in the same order, and is passed through
    unchanged. If omitted, every request gets the ``_ARRIVAL_MS_UNAVAILABLE``
    sentinel.

    Returns a 10-tuple in the exact positional order ``decide()`` expects, so
    callers can do ``rust_scheduler.decide(*flatten(...))``:

        (ids, is_context, prompt_lens, context_done_tokens, tokens_generated,
         max_new_tokens, arrival_ms, inflight, kv_free_blocks, now_ms)
    """
    from tensorrt_llm.bindings import LlmRequestState

    ids = [req.request_id for req in active_requests]
    is_context = [req.state == LlmRequestState.CONTEXT_INIT for req in active_requests]
    prompt_lens = [req.prompt_len for req in active_requests]
    context_done_tokens = [req.context_current_position for req in active_requests]
    tokens_generated = [req.get_num_tokens(0) - req.prompt_len for req in active_requests]
    max_new_tokens = [req.max_new_tokens for req in active_requests]
    if arrival_ms is None:
        arrival_ms = [_ARRIVAL_MS_UNAVAILABLE for _ in active_requests]
    inflight = list(inflight)

    return (
        ids,
        is_context,
        prompt_lens,
        context_done_tokens,
        tokens_generated,
        max_new_tokens,
        arrival_ms,
        inflight,
        kv_free_blocks,
        now_ms,
    )


def unflatten(active_requests, idx_pair):
    """
    Map a ``(fitting_indices, paused_indices)`` index pair -- as returned by
    ``RustScheduler.decide()`` -- back onto the ``LlmRequest`` objects in
    ``active_requests``, by position.

    These are INDICES into ``active_requests``, not request ids, and are
    expected to be disjoint. This function does not validate that -- it is a
    pure mapping step; ``scheduler.py`` is where a malformed index pair gets
    detected and counted before it can do damage.

    Returns ``(fitting, paused)``: lists of the original ``LlmRequest``
    objects (same object identity, never copies). ``fitting`` corresponds
    exactly to the first element of ``CapacityScheduler.schedule_request``'s
    3-tuple return -- the micro-batch scheduler one layer up is what later
    splits it into encoder/context/generation groups.
    """
    fitting_indices, paused_indices = idx_pair
    fitting = [active_requests[i] for i in fitting_indices]
    paused = [active_requests[i] for i in paused_indices]
    return fitting, paused
