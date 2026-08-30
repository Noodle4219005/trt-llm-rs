//! The pure decision logic behind [`crate::RustScheduler::decide`], kept free
//! of PyO3 types so it is testable without a Python interpreter attached (see
//! the crate-level docs for why that matters).
//!
//! ## Scope: capacity only, not the whole `RequestScheduler`
//!
//! This crate hooks `SimpleScheduler.schedule_request`'s inner
//! `capacity_scheduler.schedule_request` call, not the outer
//! `RequestScheduler` interface. `capacity_scheduler` is where the admission
//! policy -- and the measured 14% headroom -- actually lives; the sibling
//! `micro_batch_scheduler.schedule` call owns chunked-prefill chunk sizing
//! (`req.context_chunk_size`), the context/generation split, and encoder
//! requests, none of which this crate reproduces. So `decide` answers exactly
//! one question per request: admitted this step, or paused. It does not
//! distinguish context from generation in its *output* -- both land in the
//! same `fitting_indices` list -- even though the two are charged
//! differently while deciding (see below).
//!
//! `fitting_disagg_gen_init` (the third list the real
//! `capacity_scheduler.schedule_request` returns) is not produced here:
//! whether a disaggregated gen-init request fits is a KV-capacity/
//! transfer-readiness question about mechanism, not admission policy, and
//! stays on the Python side.
//!
//! ## Interpreting the flattened arrays
//!
//! `decide` receives one full, per-request snapshot each call (Python
//! flattens its request objects into arrays once, per the measured 9,533 ns
//! PyO3 round-trip cost -- see `crates/pyhost` for that measurement, and the
//! crate-level docs here for why it forces this shape). `context_done_tokens`
//! is `req.context_current_position`: with chunked prefill a context request
//! can be most of the way through its prompt already, and
//! `prompt_lens[i] - context_done_tokens[i]` -- not `prompt_lens[i]` -- is
//! the work still owed this step. `crates/sched/src/prefill.rs`'s
//! `PendingPrefill::done_tokens` plays the identical role. Charging the full
//! prompt every step would make a chunked request that is 99% done look as
//! expensive as one that has not started, and `context_done_tokens_near_prompt_len_is_charged_only_its_remainder`
//! below is the test that fails if this is ignored.

use std::collections::HashSet;

/// One step's worth of scheduler input, already validated to have
/// same-length arrays.
pub struct DecideInput<'a> {
    pub ids: &'a [i64],
    pub is_context: &'a [bool],
    pub prompt_lens: &'a [u32],
    /// `req.context_current_position` -- tokens of the prompt a chunked
    /// prefill has already computed in an earlier step. Unused for a
    /// generation-phase request.
    pub context_done_tokens: &'a [u32],
    pub arrival_ms: &'a [f64],
    pub inflight: &'a [i64],
    pub max_batch_size: usize,
    pub max_num_tokens: usize,
    /// The `ItlController`'s current concurrency cap, floored to a count.
    /// This is the crate's whole reason to exist: the cap is where the
    /// measured decode headroom (crates/sched/src/decode.rs:3-7) is actually
    /// spent. It bounds only the generation-phase candidates.
    pub generation_cap: usize,
}

/// Checks that every array `decide` receives is the same length as `ids`.
/// Returns the common length on success.
pub fn validate_lengths(ids_len: usize, lens: &[(&str, usize)]) -> Result<usize, String> {
    for (name, len) in lens {
        if *len != ids_len {
            return Err(format!(
                "decide: `{name}` has length {len}, expected {ids_len} (== `ids`.len())"
            ));
        }
    }
    Ok(ids_len)
}

/// Partitions and admits requests. Returns `(fitting_indices,
/// paused_indices)`, disjoint, all indices into the input arrays.
/// Phase-agnostic in its output: a context and a generation request that
/// both fit land in the same `fitting_indices` list.
///
/// Every admitted request -- context or generation -- charges the shared
/// `max_num_tokens` budget: a context request charges
/// `prompt_len - context_done_tokens` (the prefill work still owed this
/// step), a generation request charges exactly 1 (the one token it emits
/// this step). Generation is additionally bounded by `generation_cap`
/// (the `ItlController`'s cap); context is not.
///
/// Determinism is load-bearing here (every TP rank runs this and must
/// agree): candidates are sorted by `(arrival_ms, id)` -- never iterated out
/// of a `HashMap` -- before either budget loop runs. Generation is decided
/// first (the crate's whole reason to exist -- see module docs), then
/// context fills whatever batch room and token budget remain.
pub fn decide_indices(input: &DecideInput<'_>) -> (Vec<usize>, Vec<usize>) {
    let n = input.ids.len();
    let inflight: HashSet<i64> = input.inflight.iter().copied().collect();

    let mut context_candidates: Vec<usize> = Vec::new();
    let mut generation_candidates: Vec<usize> = Vec::new();
    for i in 0..n {
        if inflight.contains(&input.ids[i]) {
            // Scheduled by nobody this step: already running in another
            // micro-batch, so it belongs in neither list.
            continue;
        }
        if input.is_context[i] {
            context_candidates.push(i);
        } else {
            generation_candidates.push(i);
        }
    }

    let by_arrival_then_id = |ids: &[i64], arrival_ms: &[f64], a: &usize, b: &usize| {
        arrival_ms[*a]
            .total_cmp(&arrival_ms[*b])
            .then_with(|| ids[*a].cmp(&ids[*b]))
    };
    context_candidates.sort_by(|a, b| by_arrival_then_id(input.ids, input.arrival_ms, a, b));
    generation_candidates.sort_by(|a, b| by_arrival_then_id(input.ids, input.arrival_ms, a, b));

    let mut fitting_indices = Vec::new();
    let mut paused_indices = Vec::new();

    let max_num_tokens = input.max_num_tokens as u64;
    let mut batch_used = 0usize;
    let mut token_used: u64 = 0;
    let mut gen_admitted = 0usize;

    for idx in generation_candidates {
        let cost = 1u64;
        if gen_admitted < input.generation_cap
            && batch_used < input.max_batch_size
            && token_used.saturating_add(cost) <= max_num_tokens
        {
            fitting_indices.push(idx);
            gen_admitted += 1;
            batch_used += 1;
            token_used += cost;
        } else {
            paused_indices.push(idx);
        }
    }

    // A candidate that does not fit is paused, not dropped, and later
    // candidates are still tried against the remaining budget (best-effort
    // packing) rather than stopping at the first one that does not fit.
    for idx in context_candidates {
        let remaining =
            u64::from(input.prompt_lens[idx].saturating_sub(input.context_done_tokens[idx]));
        if batch_used < input.max_batch_size
            && token_used.saturating_add(remaining) <= max_num_tokens
        {
            fitting_indices.push(idx);
            batch_used += 1;
            token_used += remaining;
        } else {
            paused_indices.push(idx);
        }
    }

    (fitting_indices, paused_indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn input<'a>(
        ids: &'a [i64],
        is_context: &'a [bool],
        prompt_lens: &'a [u32],
        context_done_tokens: &'a [u32],
        arrival_ms: &'a [f64],
        inflight: &'a [i64],
        max_batch_size: usize,
        max_num_tokens: usize,
        generation_cap: usize,
    ) -> DecideInput<'a> {
        DecideInput {
            ids,
            is_context,
            prompt_lens,
            context_done_tokens,
            arrival_ms,
            inflight,
            max_batch_size,
            max_num_tokens,
            generation_cap,
        }
    }

    #[test]
    fn disjoint_and_in_range() {
        let ids = [1, 2, 3, 4];
        let is_context = [true, false, true, false];
        let prompt_lens = [100, 10, 100, 10];
        let context_done_tokens = [0, 0, 0, 0];
        let arrival_ms = [0.0, 1.0, 2.0, 3.0];
        let inflight = [];
        let inp = input(
            &ids,
            &is_context,
            &prompt_lens,
            &context_done_tokens,
            &arrival_ms,
            &inflight,
            64,
            8192,
            64,
        );
        let (fitting, paused) = decide_indices(&inp);
        for i in &fitting {
            assert!(!paused.contains(i));
        }
        let mut all: Vec<usize> = fitting.iter().chain(paused.iter()).copied().collect();
        all.sort_unstable();
        assert_eq!(all, vec![0, 1, 2, 3]);
        for i in &all {
            assert!(*i < ids.len());
        }
    }

    #[test]
    fn inflight_id_appears_in_no_list() {
        let ids = [10, 20, 30];
        let is_context = [false, false, true];
        let prompt_lens = [0, 0, 50];
        let context_done_tokens = [0, 0, 0];
        let arrival_ms = [0.0, 1.0, 2.0];
        let inflight = [20];
        let inp = input(
            &ids,
            &is_context,
            &prompt_lens,
            &context_done_tokens,
            &arrival_ms,
            &inflight,
            64,
            8192,
            64,
        );
        let (fitting, paused) = decide_indices(&inp);
        let all: Vec<usize> = fitting.into_iter().chain(paused).collect();
        assert!(
            !all.contains(&1),
            "index 1 (id 20) is inflight elsewhere and must appear nowhere: {all:?}"
        );
    }

    #[test]
    fn deterministic_across_repeated_calls() {
        let ids = [5, 1, 3, 2, 4];
        let is_context = [false, true, false, true, false];
        let prompt_lens = [0, 4000, 0, 4000, 0];
        let context_done_tokens = [0, 0, 0, 0, 0];
        let arrival_ms = [4.0, 0.0, 2.0, 1.0, 3.0];
        let inflight = [];
        let inp = input(
            &ids,
            &is_context,
            &prompt_lens,
            &context_done_tokens,
            &arrival_ms,
            &inflight,
            3,
            5000,
            2,
        );
        let first = decide_indices(&inp);
        let second = decide_indices(&inp);
        assert_eq!(first.0, second.0);
        assert_eq!(first.1, second.1);
    }

    #[test]
    fn overflow_past_max_batch_size_is_paused_not_dropped() {
        // 4 generation candidates, cap and token budget plenty, but
        // max_batch_size == 2.
        let ids = [1, 2, 3, 4];
        let is_context = [false, false, false, false];
        let prompt_lens = [0, 0, 0, 0];
        let context_done_tokens = [0, 0, 0, 0];
        let arrival_ms = [0.0, 1.0, 2.0, 3.0];
        let inflight = [];
        let inp = input(
            &ids,
            &is_context,
            &prompt_lens,
            &context_done_tokens,
            &arrival_ms,
            &inflight,
            2,
            8192,
            64,
        );
        let (fitting, paused) = decide_indices(&inp);
        assert_eq!(
            fitting.len(),
            2,
            "only max_batch_size admitted: {fitting:?}"
        );
        assert_eq!(
            paused.len(),
            2,
            "the overflow must land in paused, not vanish"
        );
        let mut all: Vec<usize> = fitting.iter().chain(paused.iter()).copied().collect();
        all.sort_unstable();
        assert_eq!(all, vec![0, 1, 2, 3]);
    }

    #[test]
    fn generation_cap_below_max_batch_size_still_pauses_the_rest() {
        let ids = [1, 2, 3];
        let is_context = [false, false, false];
        let prompt_lens = [0, 0, 0];
        let context_done_tokens = [0, 0, 0];
        let arrival_ms = [0.0, 1.0, 2.0];
        let inflight = [];
        let inp = input(
            &ids,
            &is_context,
            &prompt_lens,
            &context_done_tokens,
            &arrival_ms,
            &inflight,
            64,
            8192,
            1,
        );
        let (fitting, paused) = decide_indices(&inp);
        assert_eq!(fitting, vec![0]);
        assert_eq!(paused, vec![1, 2]);
    }

    #[test]
    fn context_respects_max_num_tokens_budget() {
        let ids = [1, 2, 3];
        let is_context = [true, true, true];
        // 3 prompts of 3000 remaining tokens each; budget only fits two.
        let prompt_lens = [3000, 3000, 3000];
        let context_done_tokens = [0, 0, 0];
        let arrival_ms = [0.0, 1.0, 2.0];
        let inflight = [];
        let inp = input(
            &ids,
            &is_context,
            &prompt_lens,
            &context_done_tokens,
            &arrival_ms,
            &inflight,
            64,
            6000,
            64,
        );
        let (fitting, paused) = decide_indices(&inp);
        assert_eq!(fitting, vec![0, 1]);
        assert_eq!(paused, vec![2]);
    }

    /// The whole reason `context_done_tokens` exists: a chunked-prefill
    /// request that is nearly finished must be charged only its remainder,
    /// not its full prompt length, against `max_num_tokens`. If this field
    /// were ignored (charging `prompt_lens[i]` alone) this request would be
    /// paused instead of admitted.
    #[test]
    fn context_done_tokens_near_prompt_len_is_charged_only_its_remainder() {
        let ids = [1];
        let is_context = [true];
        let prompt_lens = [10_000];
        let context_done_tokens = [9_990]; // only 10 tokens of prefill left
        let arrival_ms = [0.0];
        let inflight = [];
        let inp = input(
            &ids,
            &is_context,
            &prompt_lens,
            &context_done_tokens,
            &arrival_ms,
            &inflight,
            64,
            10, // budget fits the 10-token remainder, not the 10,000 full prompt
            64,
        );
        let (fitting, paused) = decide_indices(&inp);
        assert_eq!(
            fitting,
            vec![0],
            "must be admitted: charged remainder (10) fits the budget (10), \
             even though the full prompt (10,000) would not"
        );
        assert!(paused.is_empty());
    }

    /// Generation requests share the same `max_num_tokens` budget as
    /// context requests (each charges 1, for the one token it emits this
    /// step) -- this is not just a `max_batch_size` / `generation_cap`
    /// admission.
    #[test]
    fn generation_requests_also_charge_the_shared_token_budget() {
        let ids = [1, 2, 3, 4, 5];
        let is_context = [false, false, false, false, false];
        let prompt_lens = [0, 0, 0, 0, 0];
        let context_done_tokens = [0, 0, 0, 0, 0];
        let arrival_ms = [0.0, 1.0, 2.0, 3.0, 4.0];
        let inflight = [];
        // max_batch_size and generation_cap both allow all 5; only the
        // 2-token budget should bind.
        let inp = input(
            &ids,
            &is_context,
            &prompt_lens,
            &context_done_tokens,
            &arrival_ms,
            &inflight,
            64,
            2,
            64,
        );
        let (fitting, paused) = decide_indices(&inp);
        assert_eq!(fitting, vec![0, 1], "only 2 fit the 2-token budget");
        assert_eq!(paused, vec![2, 3, 4]);
    }
}
