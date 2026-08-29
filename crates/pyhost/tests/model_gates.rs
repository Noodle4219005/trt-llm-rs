//! Phase B model gates: a real TensorRT-LLM PyTorch engine, driven from Rust
//! through the [`PyHost`] boundary, on an actual GPU.
//!
//! Unlike `tests/boundary.rs`, these tests are NOT safe to run on a
//! GPU-less node: they load a real model and run real CUDA work. Each test
//! is `#[ignore]`d so the default `cargo test` run (used on any node, as a
//! build sanity check) never touches a GPU; run these explicitly with
//! `--ignored`.
//!
//! # Gates 2/3 and 8 also need the `mpirun` wrapper, not just gate 8
//!
//! `tensorrt_llm.LLM(...)` falls through, on non-Windows, to
//! `GenerationExecutor._create_ipc_executor(..., use_worker=False)` ->
//! `GenerationExecutorProxy` -> `MpiPoolSession` **even at
//! `tensor_parallel_size=1`** (see
//! `third_party/TensorRT-LLM/tensorrt_llm/executor/executor.py:638-657` and
//! `third_party/TensorRT-LLM/tensorrt_llm/llmapi/mpi_session.py:74-79`:
//! `need_spawn_mpi_workers` only gates on `model_world_size > 1`, but the
//! *un*-gated fallback at line 651 reaches `GenerationExecutorProxy`
//! regardless). `MpiPoolSession` spawns its worker via `MPI_Comm_spawn`,
//! which needs spare MPI slots to spawn into -- a bare `python3` (or a bare
//! embedded interpreter) launch dies with `MPI_ERR_SPAWN`. This matches this
//! project's own prior finding with `python3 -m dynamo.trtllm` (see wiki
//! `2026-08-29-dynamo-layer-costs-...md`, point 4: "worker 仍要包在 `mpirun`
//! 裡，即使 TP1"). So every test here that constructs a real `LLM` must run
//! under the same `mpirun --bind-to none --oversubscribe -H
//! "$(hostname):24" -n 1` wrapper the sbatch script uses for the TP2 gate --
//! this is a correction to the original task spec, which only mentioned
//! `mpirun` for TP2. Gate 4 never constructs an `LLM` (it only uses raw
//! `torch.cuda`), so it does not need the wrapper.
//!
//! # One `cargo test` invocation per gate
//!
//! The sbatch script runs each `#[ignore]`d test in this file as its own
//! `cargo test <exact-name> -- --ignored --exact` invocation (its own OS
//! process), rather than one invocation covering all of them. A `PyHost`'s
//! embedded CPython interpreter is process-global and outlives any single
//! `PyHost` (only the dedicated OS thread is torn down on `Drop`), so a
//! model loaded by one test would otherwise still be reachable -- and its
//! CUDA memory and any spawned MPI worker process still live -- when the
//! next test's `PyHost::start` runs in the same process. Separate processes
//! sidestep that entirely instead of relying on Python-side cleanup this
//! spike has not verified.

use std::time::{Duration, Instant};

use trtllm_pyhost::PyHost;

/// The model under test: overridable via `PYHOST_MODEL_PATH` (the sbatch
/// script sets it), falling back to the path named in the task brief so a
/// developer can also run this by hand from a plain shell.
fn model_path() -> String {
    std::env::var("PYHOST_MODEL_PATH").unwrap_or_else(|_| {
        "/work/u4063814/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/\
         c1899de289a04d12100db370d81485cdf75e47ca"
            .to_string()
    })
}

/// Gates 2+3: construct a real `tensorrt_llm.LLM` PyTorch-backend engine
/// from Rust through the `PyHost` boundary, run a short greedy generation,
/// and read the resulting text back into Rust.
///
/// This is the end-to-end proof for `Rust -> PyO3 -> CPython -> TRT-LLM
/// PyTorch backend -> CUDA` under an *embedded* interpreter, as opposed to
/// under a plain `python3` process.
#[tokio::test]
#[ignore]
async fn gate_2_3_llm_generates_text() {
    let host = PyHost::start(0, None).expect("PyHost::start should succeed on a GPU node");

    host.exec(&format!(
        r#"
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig

llm = LLM(
    model={model_path:?},
    backend='pytorch',
    tensor_parallel_size=1,
    kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.3),
)
"#,
        model_path = model_path(),
    ))
    .await
    .expect(
        "constructing tensorrt_llm.LLM(backend='pytorch', tensor_parallel_size=1) should \
         succeed -- if this fails with MPI_ERR_SPAWN, the mpirun wrapper did not supply enough \
         spawnable slots (see the module docs)",
    );

    host.exec(
        r#"
_gate23_sampling_params = SamplingParams(max_tokens=16, temperature=0.0)
_gate23_outputs = llm.generate(["The capital of France is"], _gate23_sampling_params)
_gate23_text = _gate23_outputs[0].outputs[0].text
"#,
    )
    .await
    .expect("llm.generate should succeed for a loaded engine");

    let text = host
        .eval_str("_gate23_text")
        .await
        .expect("reading the generated text back across the boundary should succeed");

    println!("gate 2/3 generated text: {text:?}");
    assert!(
        !text.is_empty(),
        "generated text should be non-empty, got {text:?}"
    );
}

/// Gate 4: a CUDA stream and event created in one `PyHost` crossing remain
/// valid and correctly ordered when used from later, separate crossings.
///
/// This is the gate that matters for C3: it is not enough for the boundary
/// to carry values back and forth (gates 1-3 cover that) -- CUDA objects
/// with cross-call lifetime and asynchronous completion semantics must
/// survive being handled across independent `PyHost::exec`/`eval_int` calls.
#[tokio::test]
#[ignore]
async fn gate_4_cuda_stream_event_ordering_survives_crossings() {
    let host = PyHost::start(0, None).expect("PyHost::start should succeed on a GPU node");

    // Crossing 1: create a non-default stream and event, stash them at
    // module level so a later, separate crossing can still reach them.
    host.exec(
        r#"
import torch
_gate4_stream = torch.cuda.Stream()
_gate4_event = torch.cuda.Event(enable_timing=False)
"#,
    )
    .await
    .expect("crossing 1: creating the stream and event should succeed");

    let mut matmul_iters: u32 = 200;
    let mut in_flight_observed = false;
    let mut elapsed = Duration::default();

    for attempt in 1..=3 {
        // Crossing 2: launch measurable work on that stream, then record
        // the event -- a SEPARATE crossing from crossing 1.
        let start = Instant::now();
        host.exec(&format!(
            r#"
with torch.cuda.stream(_gate4_stream):
    _gate4_a = torch.randn(4096, 4096, device='cuda')
    _gate4_b = torch.randn(4096, 4096, device='cuda')
    for _ in range({matmul_iters}):
        _gate4_c = _gate4_a @ _gate4_b
_gate4_event.record(_gate4_stream)
"#
        ))
        .await
        .expect("crossing 2: launching work on the stream and recording the event should succeed");

        // Crossing 3: query immediately. If the work already finished, this
        // test measured nothing -- retry with more work rather than pass
        // vacuously.
        let query_result = host
            .eval_int("int(_gate4_event.query())")
            .await
            .expect("crossing 3: querying the event should succeed");

        if query_result == 0 {
            in_flight_observed = true;

            // Crossing 4: synchronize, then assert completion -- again a
            // SEPARATE crossing from crossing 2 and crossing 3.
            host.exec("_gate4_event.synchronize()")
                .await
                .expect("crossing 4: synchronizing the event should succeed");
            elapsed = start.elapsed();

            let done = host
                .eval_int("int(_gate4_event.query())")
                .await
                .expect("crossing 4: querying the event after synchronize should succeed");
            assert_eq!(
                done, 1,
                "event.query() should be True after event.synchronize()"
            );
            break;
        }

        eprintln!(
            "gate 4 attempt {attempt}/3: event.query() was already True immediately after \
             record() with matmul_iters={matmul_iters}; retrying with more work"
        );
        matmul_iters *= 4;
    }

    assert!(
        in_flight_observed,
        "event.query() was True immediately after record() on all 3 attempts (up to \
         matmul_iters={matmul_iters}) -- this test measured nothing rather than proving ordering"
    );
    assert!(
        elapsed > Duration::ZERO,
        "elapsed wall time across crossings 2..4 should be greater than zero, was {elapsed:?}"
    );
    println!("gate 4 elapsed wall time across crossings 2->4: {elapsed:?}");
}

/// Gate 8: the same construction and generation as gates 2/3, but with
/// `tensor_parallel_size=2`, run under the container's `mpirun` so
/// `MpiPoolSession` has spare slots to `MPI_Comm_spawn` a second rank into.
///
/// The real question this answers: does `MPI_Comm_spawn` still work when
/// the parent interpreter is embedded in a Rust process rather than being
/// `python3` itself? A clean failure here (captured via `.expect`'s anyhow
/// context chain) is a valid, useful result for this gate, not a bug in the
/// test.
#[tokio::test]
#[ignore]
async fn gate_8_tp2_multi_rank_smoke() {
    let host = PyHost::start(0, None).expect("PyHost::start should succeed on a GPU node");

    host.exec(&format!(
        r#"
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig

llm_tp2 = LLM(
    model={model_path:?},
    backend='pytorch',
    tensor_parallel_size=2,
    kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.3),
)
"#,
        model_path = model_path(),
    ))
    .await
    .expect(
        "gate 8: constructing tensorrt_llm.LLM(tensor_parallel_size=2) failed -- see the anyhow \
         context chain above for the exact error (e.g. MPI_ERR_SPAWN if the mpirun wrapper did \
         not supply enough spawnable slots for 2 ranks); a clean failure here is a valid result",
    );

    host.exec(
        r#"
_gate8_sampling_params = SamplingParams(max_tokens=16, temperature=0.0)
_gate8_outputs = llm_tp2.generate(["The capital of France is"], _gate8_sampling_params)
_gate8_text = _gate8_outputs[0].outputs[0].text
"#,
    )
    .await
    .expect("gate 8: llm_tp2.generate should succeed after a successful TP2 construction");

    let text = host
        .eval_str("_gate8_text")
        .await
        .expect("gate 8: reading the generated text back across the boundary should succeed");

    println!("gate 8 (TP2) generated text: {text:?}");
    assert!(
        !text.is_empty(),
        "TP2 generated text should be non-empty, got {text:?}"
    );
}
