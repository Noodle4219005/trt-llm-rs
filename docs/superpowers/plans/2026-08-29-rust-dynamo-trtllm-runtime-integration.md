# Rust Dynamo + TensorRT-LLM runtime integration

## Goal

Finish the corrected Phase C and Phase D implementation so Rust is actually installed in the Dynamo + TensorRT-LLM runtime path, with tests at each boundary and no regression to upstream executor mechanisms.

## Architecture

Rust is a policy and adapter layer. `trtllm-pyext` is loaded into the TensorRT-LLM Python worker and wraps `PyExecutor.scheduler.capacity_scheduler`; the upstream scheduler remains authoritative for mechanism-validity. A feature-gated Rust crate implements Dynamo v1.4.1 `LLMEngine` and forwards request streams to a Python TRT-LLM worker. Dynamo `Worker` owns lifecycle, endpoint, health, metrics, KV events, drain, and cleanup. The old Rust `Engine` remains simulator-only.

## Tech stack

Rust workspace crates, PyO3 0.29.2 extension, Python 3 bridge, pinned TensorRT-LLM v1.3.0rc22, pinned Dynamo v1.4.1, async streams, HTTP/SSE transport, and existing `core` scoring types. No vendored source edits.

## Global constraints

- Preserve existing user changes and work in the current dirty `/home/u5727520/trt-llm-rs` checkout because the uncommitted Phase C bridge is the subject of this continuation.
- TDD: add a focused failing test before each production change; run the narrowest local checks available. Do not run cluster/GPU work without an explicit execution card and approval.
- Use only bounded, disjoint task ownership. Review every implementation task independently before accepting it.
- Keep exact paths, commands, and pass/fail criteria in the SDD progress ledger.

## Task 1 — Phase C bridge installation and safe admission

Owned files: `python/trtllm_rs_bridge/__init__.py`, `_abi.py`, `scheduler.py`, `patch.py`, `python/trtllm_rs_bridge/tests/test_bridge.py`.

First add red tests proving the exported class imports, `patch.install` changes `capacity_scheduler` only, upstream paused/disagg-init requests are preserved, and shadow mode returns upstream output. Implement the smallest compatibility fix. Rust receives only ordinary requests already accepted by upstream; it never reimplements chunked-prefill or KV feasibility. Acceptance: focused bridge tests pass and no code imports the removed `RustBackedScheduler` name.

## Task 2 — Phase C runtime feedback and Python packaging

Owned files: `crates/pyext/src/lib.rs`, `crates/pyext/src/decide.rs`, `crates/pyext/src/decide_tests.rs`, `python/trtllm_rs_bridge/patch.py`, `python/trtllm_rs_bridge/tests/test_bridge.py`, `scripts/shadow_launch.py`.

Add red tests for observer callback installation/idempotence, clean host-step fallback, overlap concurrency, malformed arrays, and stats delivery. Implement an instance-level hook around the stable PyExecutor iteration-stat update method; call `RustScheduler.observe` once per completed stat record. Keep the `decide` ABI explicit about unsupported arrival data and engine truth. Acceptance: pure Rust tests and bridge tests pass; shadow report contains nonzero observer samples under the fake executor path.

## Task 3 — Dynamo transport and request telemetry core

Owned files: new `crates/dynamo/src/{lib.rs,transport.rs,telemetry.rs}` plus crate manifest/tests.

Add failing mock-transport tests for request serialization, SSE/token conversion, one terminal event, cancellation, stream-drop cleanup, and error propagation. Implement a transport-neutral Rust adapter core and request telemetry builder. Keep the production transport behind a small trait so tests do not require a network service. Acceptance: local crate tests cover all stream terminal paths and produce `RequestOutcome`/`GoodputReport` inputs without a GPU.

## Task 4 — Pinned Dynamo v1.4.1 `LLMEngine` adapter

Owned files: `crates/dynamo/src/engine.rs`, `crates/dynamo/src/config.rs`, crate manifest, and a minimal feature-gated integration entrypoint/example.

Add red compile/type tests against `third_party/dynamo/lib/backend-common`. Implement `start`, `generate`, `abort`, `is_quiescent`, `cleanup`, health/capability/update methods, and the required conversion to `LLMEngineOutput`. Construct through Dynamo `Worker::new`, without duplicating HTTP/lifecycle/metrics/KV orchestration. Acceptance: the integration feature type-checks against the pinned submodule; default workspace builds remain independent of optional Dynamo dependencies.

## Task 5 — Scoring-aware tuner

Owned files: `crates/tuner/src/plan.rs`, `crates/tuner/src/lib.rs`, `crates/cli/src/main.rs` only if required, plus focused tests.

Add a failing fake-evaluator test where highest request rate is not highest official score. Add a measured-run evaluator seam and exact score ordering, preserving a clearly named simulation evaluator. Acceptance: `cargo test -p trtllm-tuner` (or the narrow equivalent) proves measured score wins and invalid candidates remain skipped.

## Task 6 — Integration verification and handoff

Owned files: packaging/docs/scripts only; no unrelated cleanup.

Run local focused tests, Python syntax/import checks with the intended environment when available, inspect the complete diff, and run independent review. Prepare a separate execution card for any GPU/cluster smoke; do not submit it in this task. Record durable facts in the wiki, triage inbox stubs, run `wiki check`, and sync.

## Self-review checklist

- Does the live Python hook install below `RequestScheduler` and preserve upstream mechanism decisions?
- Is overlap measured from engine-reported/runtime stats rather than assuming one token per iteration?
- Is the Dynamo adapter request-granular and cancellation-safe, with one terminal output?
- Does Dynamo `Worker`, not this project, own lifecycle and endpoint registration?
- Can default local tests run without libpython, GPU, network, or cluster access?
- Is measured tuner ranking distinct from simulator ranking and based on the exact project score?
