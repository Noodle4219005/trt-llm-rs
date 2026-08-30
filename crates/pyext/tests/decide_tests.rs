//! End-to-end tests of the public `RustScheduler` ABI (crate: `trtllm-pyext`,
//! Python module: `trtllm_rs`), called as plain Rust -- no Python interpreter
//! attached. That is exactly what `crate-type = ["cdylib", "rlib"]` buys:
//! `#[pymethods]` functions remain ordinary, directly callable Rust methods
//! as long as nothing in their body touches a live GIL, and `decide`/
//! `observe`/`stats` never do.
//!
//! `decide` is the `capacity_scheduler.schedule_request` hook, not the whole
//! `RequestScheduler` -- see the crate docs in src/lib.rs -- so it returns
//! `(fitting_indices, paused_indices)`, phase-agnostic in the output.

use trtllm_pyext::RustScheduler;

fn fitting_and_paused_are_disjoint_and_in_range(fitting: &[usize], paused: &[usize], n: usize) {
    for i in fitting {
        assert!(*i < n);
        assert!(!paused.contains(i));
    }
    for i in paused {
        assert!(*i < n);
        assert!(!fitting.contains(i));
    }
}

#[test]
fn returned_sets_are_disjoint_and_in_range() {
    let mut s = RustScheduler::new(64, 8192, 20.0, 1000);
    let (fitting, paused) = s
        .decide(
            vec![1, 2, 3, 4],
            vec![true, false, true, false],
            vec![100, 0, 200, 0],
            vec![0, 0, 0, 0],
            vec![0, 5, 50, 12],
            vec![100, 100, 100, 100],
            vec![0.0, 1.0, 2.0, 3.0],
            vec![],
            500,
            10.0,
        )
        .unwrap();
    fitting_and_paused_are_disjoint_and_in_range(&fitting, &paused, 4);
}

#[test]
fn inflight_request_is_scheduled_by_nobody() {
    let mut s = RustScheduler::new(64, 8192, 20.0, 1000);
    let (fitting, paused) = s
        .decide(
            vec![7, 8, 9],
            vec![false, false, true],
            vec![0, 0, 300],
            vec![0, 0, 0],
            vec![3, 3, 0],
            vec![100, 100, 100],
            vec![0.0, 1.0, 2.0],
            vec![8],
            500,
            10.0,
        )
        .unwrap();
    let all: Vec<usize> = fitting.into_iter().chain(paused).collect();
    assert!(
        !all.contains(&1),
        "id 8 (index 1) is inflight elsewhere and must not appear anywhere: {all:?}"
    );
}

#[test]
fn identical_inputs_give_byte_identical_output() {
    let mut s = RustScheduler::new(8, 4096, 20.0, 500);
    let call = |s: &mut RustScheduler| {
        s.decide(
            vec![5, 1, 3, 2, 4],
            vec![false, true, false, true, false],
            vec![0, 4000, 0, 4000, 0],
            vec![0, 0, 0, 0, 0],
            vec![10, 0, 20, 0, 30],
            vec![100, 100, 100, 100, 100],
            vec![4.0, 0.0, 2.0, 1.0, 3.0],
            vec![],
            200,
            1.0,
        )
        .unwrap()
    };
    let first = call(&mut s);
    let second = call(&mut s);
    assert_eq!(
        first, second,
        "same inputs must give the same decision every time"
    );
}

#[test]
fn overflowing_max_batch_size_pauses_rather_than_drops() {
    // max_batch_size = 2, but 4 generation candidates want in.
    let mut s = RustScheduler::new(2, 8192, 20.0, 1000);
    let (fitting, paused) = s
        .decide(
            vec![1, 2, 3, 4],
            vec![false, false, false, false],
            vec![0, 0, 0, 0],
            vec![0, 0, 0, 0],
            vec![1, 1, 1, 1],
            vec![100, 100, 100, 100],
            vec![0.0, 1.0, 2.0, 3.0],
            vec![],
            500,
            10.0,
        )
        .unwrap();
    assert_eq!(
        fitting.len(),
        2,
        "admitted must not exceed max_batch_size: {fitting:?}"
    );
    assert_eq!(
        paused.len(),
        2,
        "the other two must be paused, not dropped: {paused:?}"
    );
}

#[test]
fn observe_moving_the_controller_changes_generation_admission() {
    // A tight cap: start the controller low and keep it there by reporting
    // step latency well over budget with a saturated batch, so the AIMD
    // cap decreases instead of climbing back to max_batch_size.
    let mut s = RustScheduler::new(64, 8192, 20.0, 1000);

    let ids: Vec<i64> = (0..20).collect();
    let is_context = vec![false; 20];
    let prompt_lens = vec![0u32; 20];
    let context_done_tokens = vec![0u32; 20];
    let tokens_generated = vec![1u32; 20];
    let max_new_tokens = vec![100u32; 20];
    let arrival_ms: Vec<f64> = (0..20).map(|i| i as f64).collect();

    let decide_now = |s: &mut RustScheduler| {
        s.decide(
            ids.clone(),
            is_context.clone(),
            prompt_lens.clone(),
            context_done_tokens.clone(),
            tokens_generated.clone(),
            max_new_tokens.clone(),
            arrival_ms.clone(),
            vec![],
            500,
            0.0,
        )
        .unwrap()
    };

    let (fitting_before, _) = decide_now(&mut s);
    assert_eq!(
        fitting_before.len(),
        20,
        "with a fresh, un-observed controller the cap starts at max_batch_size"
    );

    // Feed enough slow steps that the AIMD controller backs the cap off well
    // below 20. ItlController::observe ignores the first 8 samples, then
    // multiplies the cap by 0.9 on every later call while ewma_ms (pinned at
    // 40.0 here) stays above target_ms * high_water (19.6).
    //
    // This comment used to end "and would in fact walk the cap all the way down
    // to its min_cap = 1.0 floor" -- describing, approvingly, the pathology that
    // produced goodput 0.00 on Qwen3-235B (job 314882). Latency pinned at 40 ms
    // no matter how far the cap falls is precisely the case where falling does
    // not help, so the controller now gives up steering after one probe window
    // instead of collapsing. The cap still falls, which is what this test is
    // about; it just stops before reaching the floor.
    // Drive until the cap is actually below the request count, which is what
    // makes admission visibly shrink -- the controller starts at
    // max_batch_size = 64, so a fixed small number of observations would leave
    // it above 20 and prove nothing. Bounded by the controller giving up: with
    // latency pinned at 40 ms no matter the cap, it is entitled to conclude
    // concurrency is not the lever and restore the cap, and that is a different
    // contract, tested separately in crates/sched.
    for _ in 0..64 {
        if s.stats()["cap"] < 20.0 || s.stats()["concurrency_not_binding"] > 0.0 {
            break;
        }
        s.observe(40.0, 20);
    }
    assert!(
        s.stats()["concurrency_not_binding"] == 0.0,
        "this test is about the throttling phase; the controller should not have \
         given up yet (cap = {})",
        s.stats()["cap"]
    );

    let (fitting_after, paused_after) = decide_now(&mut s);
    assert!(
        fitting_after.len() < fitting_before.len(),
        "observe() reporting sustained over-budget latency must shrink admission: before {}, after {}",
        fitting_before.len(),
        fitting_after.len()
    );
    assert_eq!(fitting_after.len() + paused_after.len(), 20);
}

#[test]
fn stats_reports_the_expected_keys() {
    let mut s = RustScheduler::new(64, 8192, 20.0, 1000);
    let _ = s
        .decide(
            vec![1],
            vec![false],
            vec![0],
            vec![0],
            vec![1],
            vec![100],
            vec![0.0],
            vec![],
            500,
            0.0,
        )
        .unwrap();
    s.observe(15.0, 1);
    let stats = s.stats();
    for key in [
        "admitted",
        "refused",
        "cap",
        "finish_disagreements",
        "observed_itl_ms",
        "samples",
        "steps_observed",
        "kv_total_blocks",
        "kv_free_blocks_last",
        "last_now_ms",
    ] {
        assert!(
            stats.contains_key(key),
            "stats() missing key `{key}`: {stats:?}"
        );
    }
    assert_eq!(stats["admitted"], 1.0);
    assert_eq!(stats["steps_observed"], 1.0);
    assert_eq!(stats["kv_total_blocks"], 1000.0);
}

#[test]
fn mismatched_array_lengths_are_rejected() {
    let mut s = RustScheduler::new(64, 8192, 20.0, 1000);
    let result = s.decide(
        vec![1, 2],
        vec![true], // wrong length
        vec![100, 100],
        vec![0, 0],
        vec![0, 0],
        vec![100, 100],
        vec![0.0, 0.0],
        vec![],
        500,
        0.0,
    );
    assert!(result.is_err());
}

/// The whole reason `context_done_tokens` exists, exercised through the
/// public `RustScheduler` ABI (not just the pure decision function in
/// src/decide.rs): a chunked-prefill request nearly finished must be
/// charged only its remainder against `max_num_tokens`, not its full
/// prompt length. If `context_done_tokens` were ignored, this request
/// would be paused instead of admitted.
#[test]
fn context_done_tokens_near_prompt_len_is_charged_only_its_remainder() {
    let mut s = RustScheduler::new(64, 10, 20.0, 1000); // max_num_tokens = 10
    let (fitting, paused) = s
        .decide(
            vec![1],
            vec![true],
            vec![10_000], // full prompt length
            vec![9_990],  // already computed by an earlier chunk
            vec![0],
            vec![100],
            vec![0.0],
            vec![],
            500,
            0.0,
        )
        .unwrap();
    assert_eq!(
        fitting,
        vec![0],
        "remainder (10) must fit the 10-token budget even though the full prompt (10,000) would not"
    );
    assert!(paused.is_empty());
}
