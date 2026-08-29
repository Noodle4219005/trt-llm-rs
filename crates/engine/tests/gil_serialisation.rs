//! P2 — why this deployment is one OS process per worker.
//!
//! `crates/worker/src/deployment.rs:88,119` spawns N prefill + M decode engines as tokio tasks
//! inside ONE process. That is correct for `MockEngine`, which touches no Python. It is wrong for
//! any engine that drives torch: `libtensorrt_llm.so` has `libpython3.12.so.1.0` in its NEEDED list
//! and 98 undefined `Py*` symbols, so every real engine needs a thread that holds the GIL for the
//! duration of a forward pass. Five such threads in one address space contend for ONE lock.
//!
//! This test does not measure the real GIL — it measures the arithmetic of a single global lock,
//! which is what a GIL is. The real per-step boundary cost is Phase B gate 6, on hardware.
//!
//! The conclusion it exists to justify: the 4P1D topology must be 5 processes, not 5 tasks.
//! `WorkerState::endpoint` is already a string (`crates/router/src/registry.rs`), so the router was
//! built for that; `Deployment::spawn` was not.

use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Stand-in for the CPython GIL: one holder at a time, process-wide.
struct FakeGil(Mutex<()>);

/// Each "engine" runs `steps` forward passes. A forward pass holds the GIL for `hold`.
/// Returns the mean observed step latency, which is what `ItlController` would see.
fn mean_step_latency(engines: usize, steps: usize, hold: Duration) -> Duration {
    let gil = Arc::new(FakeGil(Mutex::new(())));
    let start = Arc::new(Barrier::new(engines));
    let handles: Vec<_> = (0..engines)
        .map(|_| {
            let gil = Arc::clone(&gil);
            let start = Arc::clone(&start);
            thread::spawn(move || {
                start.wait();
                let mut total = Duration::ZERO;
                for _ in 0..steps {
                    let t0 = Instant::now();
                    let _held = gil.0.lock().expect("gil poisoned");
                    thread::sleep(hold); // torch forward: GIL held throughout
                    total += t0.elapsed();
                }
                total / steps as u32
            })
        })
        .collect();
    let means: Vec<Duration> = handles
        .into_iter()
        .map(|h| h.join().expect("thread"))
        .collect();
    means.iter().sum::<Duration>() / means.len() as u32
}

/// One engine per process sees only its own forward pass.
#[test]
fn a_single_engine_sees_only_its_own_forward() {
    let hold = Duration::from_millis(15);
    let observed = mean_step_latency(1, 8, hold);
    assert!(
        observed >= hold && observed < hold * 2,
        "one engine should see ~15 ms, saw {observed:?}"
    );
}

/// Five engines in one process serialise: each step waits behind the other four.
/// This is the 4P1D shape, and it is why it cannot live in one process.
#[test]
fn five_engines_in_one_process_serialise() {
    let hold = Duration::from_millis(15);
    let solo = mean_step_latency(1, 8, hold);
    let shared = mean_step_latency(5, 8, hold);

    // 2x, not 5x. The claim under test is "they serialise rather than overlap",
    // and 2x already refutes overlap; the exact multiple is a property of the
    // machine, not of the GIL. Job 312868 failed this at 3x by SIX MICROSECONDS
    // (solo 15.059511ms, shared 45.172449ms = 2.9996x) -- a threshold sitting on
    // top of the measured value turns a real invariant into a coin flip, and a
    // build that flakes gets its failures ignored, which is worse than no test.
    assert!(
        shared > solo * 2,
        "5 engines sharing one GIL must serialise: solo {solo:?} vs shared {shared:?} \
         (anything near 1x would mean this test is not measuring contention)"
    );
    // Sanity on the other side: fully serial plus generous slack. Loose on purpose --
    // a loaded node legitimately produces a higher ratio, and this bound exists only
    // to catch a measurement that has stopped measuring anything.
    assert!(
        shared < solo * 12,
        "shared {shared:?} is more than 12x solo {solo:?}: the harness, not the GIL"
    );
}
