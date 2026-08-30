# Rust control plane in the Dynamo + TensorRT-LLM runtime

## Goal

Complete Phase C and Phase D so Rust is installed in the real Dynamo + TensorRT-LLM serving path. Rust owns policy, lifecycle-facing adapter code, request telemetry, and tuning decisions; TensorRT-LLM retains model execution, sampling, CUDA graphs, overlap scheduling, KV allocation, and KV transport.

## Architecture

There are two runtime boundaries:

1. `trtllm-pyext` is loaded by each TensorRT-LLM Python worker. `RustBackedCapacityScheduler` is installed at `PyExecutor.scheduler.capacity_scheduler`, below `RequestScheduler` and above TensorRT-LLM's `MicroBatchScheduler`. The upstream capacity scheduler remains authoritative for KV feasibility, prefix reuse, PEFT, encoder requests, chunked prefill, and disaggregated generation initialization. Rust may apply a conservative admission policy to ordinary upstream-fitting requests, but it must not reproduce micro-batch or KV mechanisms.
2. A Rust Dynamo backend implements the pinned v1.4.1 `dynamo_backend_common::LLMEngine` contract. It forwards request-granularity generation to a TensorRT-LLM Python worker through an explicit streaming transport, maps terminal events and cancellation, and lets Dynamo `Worker` own registration, health, metrics, KV-event publication, drain, and cleanup.

The existing `crates/engine::Engine` trait remains simulator-only. It is not used as the serving adapter and no legacy C++ TensorRT-LLM FFI is restored.

## Phase C deliverables

- Correct class/module exports and idempotent installation at the capacity-scheduler field.
- A tested ABI with deterministic request IDs, explicit context progress, and no fabricated arrival-time claim.
- Conservative live policy: upstream mechanism selection first; Rust only orders/selects ordinary fitting candidates; upstream paused and disaggregated-init requests pass through unchanged.
- A runtime observer attached to the existing PyExecutor iteration-stat update seam. It records the clean host step latency when available, falls back to the reported iteration latency, and reports the scheduled ordinary concurrency to `RustScheduler.observe`.
- Shadow mode compares the same candidate set and returns upstream output. Live mode is opt-in and has an installation marker.
- Tests cover import/export, installation target, upstream pass-through, disagg pass-through, empty input, malformed ABI, deterministic ordering, observer idempotence, and overlap-safe feedback.

## Phase D deliverables

- A feature-gated Rust Dynamo integration crate/module using the pinned v1.4.1 `LLMEngine` trait and `Worker` lifecycle.
- A request-stream transport abstraction with a production HTTP/SSE implementation and deterministic mock transport tests. The adapter must emit one terminal output, poll cancellation, release per-request state on every stream-drop path, and make `cleanup` idempotent.
- Explicit start metadata, health payload, metrics hooks, and capability/update reporting. Unsupported controls return a typed Dynamo error rather than pretending to be implemented.
- End-to-end request telemetry sufficient to construct `RequestOutcome` and `GoodputReport` from arrival, first-token, token, terminal, and SLO data.
- A tuner evaluator seam that ranks measured short runs by the exact score used by the project, while retaining simulation as a separately named planning mode. A fake evaluator test proves ranking does not silently fall back to raw request rate.

## Constraints and non-goals

- Do not edit either vendored submodule unless a separately reviewed upstream compatibility fix is unavoidable.
- Do not make the Python executor loop Rust-owned; that would break overlap, speculative decoding, CUDA graph, and KV lifecycle semantics.
- Do not claim GPU/cluster runtime validation without a current execution card and explicit user approval. Local unit, packaging, mock-transport, and source-level integration checks are still required.
- Do not use the simulator `Engine` as evidence that the Dynamo serving path is integrated.

## Acceptance gates

Phase C passes when the extension imports in the intended TensorRT-LLM environment, installation mutates only `capacity_scheduler`, the focused bridge tests pass, and observer telemetry is exercised by a fake PyExecutor iteration-stat callback.

Phase D passes locally when the adapter compiles behind its feature gate, mock streaming tests cover normal completion/cancel/drop/error, Dynamo contract construction is type-checked against v1.4.1, and tuner tests rank by measured score. A real GPU smoke is a separate authorized gate.
