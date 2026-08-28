# What replaces what

This project replaces the Python control plane of Dynamo + TensorRT-LLM. It does
not replace CUDA kernels, TensorRT engines, NCCL, or the model implementations.

## Dynamo Python components

| Dynamo (`components/src/dynamo/...`) | Here | Notes |
|---|---|---|
| `frontend/` — OpenAI HTTP, SSE | `crates/frontend` | Same wire format. Arrival timestamp is stamped at accept, never later. |
| `router/` — KV-aware routing | `crates/router` | Cost is predicted TTFT in milliseconds, not a tuned affinity bonus. |
| `trtllm/main.py`, `workers/llm_worker.py` | `crates/worker` | One async task per worker around an `Engine`. |
| `trtllm/request_handlers/*` | `crates/worker/src/{prefill,decode}_worker.rs` | Prefill and decode loops are separate types, not a shared handler with mode flags. |
| `trtllm/publisher.py`, `metrics.py` | `crates/worker` `Deployment::snapshot` | |
| `trtllm/utils/disagg_utils.py` | `crates/transfer` | Reshard mapping is a tested value, not an inline index computation. |
| `planner/`, `profiler/` | `crates/sim` + `crates/tuner` | Planning happens in simulation against the scored metric, before deployment. |
| `mocker/` | `crates/engine` `MockEngine` | Same idea: run the control plane without GPUs. |

## TensorRT-LLM Python

| `tensorrt_llm` Python | Here | Notes |
|---|---|---|
| PyExecutor scheduling loop | `crates/sched` | Where the actual policy changes live. |
| Capacity / micro-batch scheduler | `crates/sched/src/prefill.rs`, `decode.rs` | Deadline-feasible batching and AIMD admission replace fixed caps. |
| KV cache manager (block pool, reuse) | `crates/kvcache` | Paged pool + hash-chain prefix cache. |
| Disaggregated serving orchestration | `crates/worker` + `crates/transfer` | |
| `Executor` C++ API bindings | `crates/engine` feature `trtllm` | **Not built in this tree** — see `trtllm-ffi.md`. |
| Kernels, plugins, TRT engines, NCCL | unchanged | Reached through the FFI seam. |

## What deliberately has no Rust equivalent

- Model definitions and quantisation. Those belong to the backend.
- Multimodal encode paths. Out of scope for this workload.
- Kubernetes/DGD manifest generation. AIConfigurator already emits those.
