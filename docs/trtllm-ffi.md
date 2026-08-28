# The TensorRT-LLM seam

`crates/engine` feature `trtllm`. **Not built or tested in this tree**: there is
no CUDA toolchain and no TensorRT-LLM install here, and shipping a binding that
has never been compiled as if it worked would be worse than shipping none.

## Integration point

`tensorrt_llm::executor::Executor` — the C++ API underneath the Python
`tensorrt_llm.LLM` class. It is the right seam because:

- it already exposes `enqueueRequest` / `awaitResponses`, which maps onto
  `Engine::prefill` and `Engine::decode_step` without inventing a protocol;
- it supports disaggregated serving directly through `ContextPhaseParams`, so
  the prefill/decode split does not have to be re-implemented above it;
- everything below it — kernels, plugins, NCCL, the engine build — stays
  untouched, which is the entire point of the rewrite.

## Shape of the shim

C++ cannot be called from Rust directly, so a narrow `extern "C"` shim is needed
(or `cxx`, which generates one). Keep it narrow: every type that crosses is a
type that has to be kept in sync.

```c
// trtllm_shim.h
typedef struct TrtllmExecutor TrtllmExecutor;

TrtllmExecutor* trtllm_executor_create(const char* engine_dir,
                                       const char* json_config,
                                       char* err, size_t err_len);
void            trtllm_executor_destroy(TrtllmExecutor*);

// Returns a request id, or 0 on failure.
uint64_t trtllm_enqueue(TrtllmExecutor*, const uint32_t* tokens, size_t n,
                        const char* sampling_json,
                        const char* context_phase_json /* NULL for aggregated */);

// Fills `out` with up to `cap` responses; returns how many were written.
size_t trtllm_await(TrtllmExecutor*, TrtllmResponse* out, size_t cap,
                    uint64_t timeout_ms);

void trtllm_cancel(TrtllmExecutor*, uint64_t request_id);
```

`build.rs` compiles the shim and links `libtensorrt_llm.so`, gated on the
feature so the default build stays pure Rust.

## Things that will bite

- **Elapsed time must be measured, not estimated.** `PrefillOutcome::elapsed_ms`
  feeds the prefill rate EWMA, which feeds every deadline decision. A plausible
  constant there corrupts scheduling silently.
- **The executor has its own scheduler.** Configure it to hand back control —
  fixed batch, no internal reordering — or two schedulers will fight and the
  admission control here becomes advisory.
- **KV block ids are ours, not the executor's.** Either hand the executor a
  pre-allocated pool it must use, or delete `crates/kvcache` from the decode
  path and read its accounting instead. Two allocators over one arena is the
  bug that shows up as a memory error hours in.
- **The first token is sampled on the prefill worker.** If the executor does not
  return it from the context phase, TTFT silently becomes prefill + transfer +
  one decode step, which is a different metric from the one being scored.
