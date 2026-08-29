//! Boundary tests for the Rust <-> embedded CPython crossing.
//!
//! These run WITHOUT a GPU -- `PyHost::start` skips `torch.cuda.set_device`
//! when CUDA is unavailable -- but they DO need the container's Python (torch
//! + tensorrt_llm importable). Run inside
//! `tensorrtllm-runtime-1.4.1.sif` with `--test-threads=1`: there is one
//! embedded interpreter per process, so tests must not race each other for
//! it.

use std::time::{Duration, Instant};

use trtllm_pyhost::PyHost;

#[tokio::test]
async fn starts_and_reports_a_trtllm_version() {
    let host = PyHost::start(0, None).expect("PyHost::start should succeed inside the container");

    let version = host
        .trtllm_version()
        .expect("trtllm_version() should be Some after a successful start");
    assert!(
        version.starts_with("1.3.0rc22"),
        "expected tensorrt_llm.__version__ to start with 1.3.0rc22, got {version:?}"
    );
}

#[tokio::test]
async fn a_python_exception_becomes_a_rust_error_without_poisoning() {
    let host = PyHost::start(0, None).expect("PyHost::start should succeed inside the container");

    let err = host
        .eval_int("1/0")
        .await
        .expect_err("1/0 should raise ZeroDivisionError, not return Ok");
    let root_cause = err.root_cause().to_string();
    assert!(
        root_cause.contains("ZeroDivisionError"),
        "expected the root cause to mention ZeroDivisionError, got: {root_cause}"
    );

    // The point of this test: one bad call must not kill the host. A worker
    // thread that panicked or got stuck holding the GIL would make this
    // second call hang or error instead of returning 42.
    let ok = host
        .eval_int("6*7")
        .await
        .expect("a later call must not be poisoned by the earlier ZeroDivisionError");
    assert_eq!(ok, 42);
}

#[tokio::test]
async fn shutdown_does_not_deadlock() {
    // Backstop against a real hang: if `drop(host)` never returns, the test
    // would otherwise hang forever rather than fail. `recv_timeout` (not a
    // plain sleep) lets the watchdog stand down the instant shutdown
    // completes, so it cannot fire late and abort an unrelated later test.
    let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
    let watchdog = std::thread::spawn(move || {
        if done_rx.recv_timeout(Duration::from_secs(10)).is_err() {
            eprintln!("shutdown_does_not_deadlock: PyHost::drop did not return within 10s");
            std::process::abort();
        }
    });

    let host = PyHost::start(0, None).expect("PyHost::start should succeed inside the container");
    host.ping().await.expect("ping before shutdown");
    host.eval_int("1 + 1").await.expect("eval before shutdown");

    let start = Instant::now();
    drop(host);
    let elapsed = start.elapsed();

    let _ = done_tx.send(());
    watchdog.join().expect("watchdog thread panicked");

    assert!(
        elapsed < Duration::from_secs(10),
        "PyHost::drop took {elapsed:?}, expected well under 10s"
    );
}

#[tokio::test]
async fn per_crossing_overhead_is_measured_and_reported() {
    let host = PyHost::start(0, None).expect("PyHost::start should succeed inside the container");

    let overhead_ns = host
        .crossing_overhead_ns()
        .await
        .expect("crossing_overhead_ns should complete 10,000 ping round trips");

    // This is a MEASUREMENT, not a correctness check: print it prominently
    // for a human to read as a go/no-go gate, and assert only a loose sanity
    // bound so this test does not flake on a loaded node.
    println!("mean ns per crossing: {overhead_ns}");
    assert!(
        overhead_ns < 1_000_000.0,
        "per-crossing overhead should be well under 1ms, was {overhead_ns} ns"
    );
}
