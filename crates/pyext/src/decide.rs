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

/// Which running request gives up its step when the concurrency cap binds.
///
/// Pausing a generation request is this crate's only lever over a request that
/// has already started, so which one is paused is the whole of the decode
/// policy. Two stacks answer it differently and neither uses the criterion
/// being scored: vLLM preempts the most recent arrival, SGLang retracts under
/// memory pressure. Both are about the machine's state rather than about what
/// pausing costs the request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DecodeOrder {
    /// Arrival, then id. What this crate did first, and what the reference
    /// scheduler does.
    ///
    /// It degenerates on exactly this benchmark. AIPerf opens with every
    /// request at t=0, so every arrival is equal and the tiebreak is the id --
    /// which makes the choice of victim arbitrary at the one moment the pool
    /// is most oversubscribed.
    Arrival,
    /// The request that can most afford to stall goes first.
    ///
    /// `good` requires mean ITL <= budget over the whole request, so a
    /// sequence that has emitted 190 of 200 tokens at a mean of 15 ms can
    /// absorb 125 ms per remaining token and still pass, while one that has
    /// emitted 5 at the same mean can absorb 20.03. Pausing the first costs
    /// almost nothing and pausing the second is most of a failed request.
    ///
    /// This is `RunningSeq::tolerable_itl_ms`, which the planner has computed
    /// since the beginning and the live path did not read.
    #[default]
    Slack,
}

/// The per-token latency a sequence can still sustain and finish inside
/// `budget_ms`, mirroring `trtllm_sched::RunningSeq::tolerable_itl_ms`.
///
/// `elapsed` is measured from when this worker first saw the request. Under
/// disaggregation that is when it arrived from the prefill worker, whose
/// sampling produced token one -- so `tokens_generated - 1` gaps have been
/// observed over that window, which is the quantity the criterion scores.
fn tolerable_itl_ms(
    now_ms: f64,
    arrival_ms: f64,
    tokens_generated: u32,
    max_new_tokens: u32,
    budget_ms: f64,
) -> f64 {
    let gaps_total = f64::from(max_new_tokens.saturating_sub(1));
    let gaps_done = f64::from(tokens_generated.saturating_sub(1));
    let remaining = gaps_total - gaps_done;
    if remaining <= 0.0 {
        // Nothing left to lose: it can stall indefinitely without changing a
        // mean it is no longer accumulating.
        return f64::INFINITY;
    }
    let elapsed = (now_ms - arrival_ms).max(0.0);
    (budget_ms * gaps_total - elapsed) / remaining
}

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
    /// Tokens this request has produced, prefill's first token included.
    pub tokens_generated: &'a [u32],
    /// `req.max_new_tokens`.
    pub max_new_tokens: &'a [u32],
    /// Monotonic now, same clock as `arrival_ms`.
    pub now_ms: f64,
    /// Mean-ITL budget from the scored criterion, milliseconds.
    pub itl_budget_ms: f64,
    /// How to choose which generation requests stall when the cap binds.
    pub decode_order: DecodeOrder,
    /// Free KV blocks reported by the engine this step.
    pub kv_free_blocks: usize,
    /// Free blocks below which a request that has not started is not admitted.
    ///
    /// vLLM applies its watermark to WAITING and PREEMPTED requests only and
    /// never to running ones (kv_cache_manager.py:462-469): the pool protects
    /// work in progress from work that has not begun. The same asymmetry
    /// applies here, with "has not begun" meaning a context request whose
    /// `context_done_tokens` is still zero -- a chunked prefill already
    /// underway is running work and is not held back.
    pub kv_watermark_blocks: usize,
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
    // Context stays on arrival: a prompt that has not started has no ITL to
    // reason about, and every prefill candidate in this benchmark's opening
    // burst is identical anyway.
    context_candidates.sort_by(|a, b| by_arrival_then_id(input.ids, input.arrival_ms, a, b));

    match input.decode_order {
        DecodeOrder::Arrival => {
            generation_candidates
                .sort_by(|a, b| by_arrival_then_id(input.ids, input.arrival_ms, a, b));
        }
        DecodeOrder::Slack => {
            // Least tolerance first, so the requests admitted are the ones that
            // cannot afford to wait and the ones paused are the ones that can.
            // Ties break on id, so the decision stays deterministic -- a
            // scheduler whose output depends on hash order cannot be debugged
            // from a log.
            generation_candidates.sort_by(|a, b| {
                let t = |i: &usize| {
                    tolerable_itl_ms(
                        input.now_ms,
                        input.arrival_ms[*i],
                        input.tokens_generated[*i],
                        input.max_new_tokens[*i],
                        input.itl_budget_ms,
                    )
                };
                t(a).total_cmp(&t(b))
                    .then_with(|| input.ids[*a].cmp(&input.ids[*b]))
            });
        }
    }

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
    // If the cap forced a running sequence to stall, do not start new work in
    // the same step. vLLM does this unconditionally (scheduler.py:775: `if not
    // preempted_reqs`), and the asymmetry in this benchmark's budgets makes it
    // more clearly right here than there -- TTFT has 3000 ms of room and mean
    // ITL has 20, so delaying a prefill is nearly free while stalling a decode
    // is most of a failed request.
    //
    // Under disaggregation this is inert by construction: a prefill worker
    // sees no generation requests and a decode worker sees no context ones, so
    // the two lists are never both non-empty in the same scheduler. It fires
    // only in the aggregated case, which is the case it was written for.
    let stalled_a_running_sequence = !paused_indices.is_empty();

    for idx in context_candidates {
        if stalled_a_running_sequence {
            paused_indices.push(idx);
            continue;
        }
        // The watermark, on requests that have not started only.
        if input.context_done_tokens[idx] == 0 && input.kv_free_blocks < input.kv_watermark_blocks {
            paused_indices.push(idx);
            continue;
        }
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
        // These existing tests predate the slack ordering and are about the
        // budgets, so they run on arrival order and feed the slack inputs
        // values that make every sequence identical -- the ordering must not
        // change what they assert.
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
            tokens_generated: &[],
            kv_free_blocks: usize::MAX,
            kv_watermark_blocks: 0,
            max_new_tokens: &[],
            now_ms: 0.0,
            itl_budget_ms: 20.0,
            decode_order: DecodeOrder::Arrival,
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

#[cfg(test)]
mod slack_tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn input<'a>(
        ids: &'a [i64],
        gen: &'a [u32],
        maxnew: &'a [u32],
        arrival: &'a [f64],
        is_ctx: &'a [bool],
        plen: &'a [u32],
        done: &'a [u32],
        cap: usize,
        order: DecodeOrder,
    ) -> DecideInput<'a> {
        DecideInput {
            ids,
            is_context: is_ctx,
            prompt_lens: plen,
            context_done_tokens: done,
            arrival_ms: arrival,
            inflight: &[],
            max_batch_size: 64,
            max_num_tokens: 16384,
            generation_cap: cap,
            tokens_generated: gen,
            max_new_tokens: maxnew,
            now_ms: 3000.0,
            itl_budget_ms: 20.0,
            decode_order: order,
            kv_free_blocks: usize::MAX,
            kv_watermark_blocks: 0,
        }
    }

    /// The case the criterion makes and arrival order cannot see.
    ///
    /// Two requests, both arrived at t=0, both at a mean ITL of 15 ms. One has
    /// emitted 190 of 200 tokens and can absorb 125 ms per remaining token;
    /// the other has emitted 5 and can absorb 20.03. Room for one. Pausing the
    /// first costs nothing scorable and pausing the second is most of a failed
    /// request -- and to arrival order they are indistinguishable.
    #[test]
    fn the_sequence_that_can_afford_to_wait_is_the_one_that_waits() {
        let ids = [1i64, 2];
        let gen = [191u32, 6]; // 190 and 5 gaps observed
        let maxnew = [200u32, 200];
        let arrival = [0.0f64, 0.0];
        let is_ctx = [false, false];
        let plen = [4000u32, 4000];
        let done = [4000u32, 4000];

        let i = input(
            &ids,
            &gen,
            &maxnew,
            &arrival,
            &is_ctx,
            &plen,
            &done,
            1,
            DecodeOrder::Slack,
        );
        let (fitting, paused) = decide_indices(&i);
        assert_eq!(fitting, vec![1], "the young sequence must keep running");
        assert_eq!(paused, vec![0], "the nearly-finished one can wait");

        // Arrival order cannot tell them apart, and picks by id.
        let i = input(
            &ids,
            &gen,
            &maxnew,
            &arrival,
            &is_ctx,
            &plen,
            &done,
            1,
            DecodeOrder::Arrival,
        );
        let (fitting, _) = decide_indices(&i);
        assert_eq!(
            fitting,
            vec![0],
            "arrival order should pause the sequence with 20 ms of tolerance, \
             which is the point of this comparison"
        );
    }

    /// A sequence with nothing left to emit cannot have its mean changed, so
    /// it must sort last rather than compete for a slot.
    #[test]
    fn a_finished_sequence_never_displaces_a_running_one() {
        let ids = [1i64, 2];
        let gen = [200u32, 6];
        let maxnew = [200u32, 200];
        let arrival = [0.0f64, 0.0];
        let is_ctx = [false, false];
        let plen = [4000u32, 4000];
        let done = [4000u32, 4000];
        let i = input(
            &ids,
            &gen,
            &maxnew,
            &arrival,
            &is_ctx,
            &plen,
            &done,
            1,
            DecodeOrder::Slack,
        );
        let (fitting, paused) = decide_indices(&i);
        assert_eq!(fitting, vec![1]);
        assert_eq!(paused, vec![0]);
    }

    /// A sequence already past its budget has negative tolerance and must be
    /// first, not written off. The benchmark is closed-loop: a request that
    /// stalls does not vanish, it holds a client thread and stops new work
    /// arriving, so abandoning it costs throughput as well as its own score.
    #[test]
    fn a_sequence_already_over_budget_is_served_first() {
        let ids = [1i64, 2, 3];
        // id 2 has burned 3000 ms over 5 gaps: 600 ms/token, far past 20.
        let gen = [6u32, 6, 6];
        let maxnew = [200u32, 200, 200];
        let arrival = [2900.0f64, 0.0, 2900.0];
        let is_ctx = [false, false, false];
        let plen = [4000u32, 4000, 4000];
        let done = [4000u32, 4000, 4000];
        let i = input(
            &ids,
            &gen,
            &maxnew,
            &arrival,
            &is_ctx,
            &plen,
            &done,
            1,
            DecodeOrder::Slack,
        );
        let (fitting, _) = decide_indices(&i);
        assert_eq!(fitting, vec![1], "the one in trouble runs");
    }

    /// Ordering must not depend on iteration order. A scheduler whose output
    /// moves with a hash cannot be debugged from a log.
    #[test]
    fn ties_are_broken_deterministically() {
        let ids = [7i64, 3, 5];
        let gen = [6u32, 6, 6];
        let maxnew = [200u32, 200, 200];
        let arrival = [0.0f64, 0.0, 0.0];
        let is_ctx = [false, false, false];
        let plen = [4000u32, 4000, 4000];
        let done = [4000u32, 4000, 4000];
        let i = input(
            &ids,
            &gen,
            &maxnew,
            &arrival,
            &is_ctx,
            &plen,
            &done,
            2,
            DecodeOrder::Slack,
        );
        let (fitting, _) = decide_indices(&i);
        // ids 3 and 5 -- the two smallest -- at indices 1 and 2.
        assert_eq!(fitting, vec![1, 2]);
    }
}

#[cfg(test)]
mod backpressure_tests {
    use super::*;
    #[allow(clippy::too_many_arguments)]
    fn mixed<'a>(
        ids: &'a [i64],
        is_ctx: &'a [bool],
        plen: &'a [u32],
        done: &'a [u32],
        gen: &'a [u32],
        maxnew: &'a [u32],
        arrival: &'a [f64],
        cap: usize,
        free: usize,
        watermark: usize,
    ) -> DecideInput<'a> {
        DecideInput {
            ids,
            is_context: is_ctx,
            prompt_lens: plen,
            context_done_tokens: done,
            arrival_ms: arrival,
            inflight: &[],
            max_batch_size: 64,
            max_num_tokens: 65536,
            generation_cap: cap,
            tokens_generated: gen,
            max_new_tokens: maxnew,
            now_ms: 1000.0,
            itl_budget_ms: 20.0,
            decode_order: DecodeOrder::Slack,
            kv_free_blocks: free,
            kv_watermark_blocks: watermark,
        }
    }

    /// Starting new work in a step that already stalled running work makes the
    /// stall worse. vLLM refuses it outright (scheduler.py:775) and the budget
    /// asymmetry here is larger: TTFT has 3000 ms of room, mean ITL has 20.
    #[test]
    fn a_step_that_stalls_a_decode_does_not_start_a_prefill() {
        let ids = [1i64, 2, 3];
        let is_ctx = [false, false, true];
        let plen = [4000u32, 4000, 4000];
        let done = [4000u32, 4000, 0];
        let gen = [6u32, 6, 0];
        let maxnew = [200u32, 200, 200];
        let arrival = [0.0f64, 0.0, 0.0];

        // Cap 1: one of the two decodes stalls, so the prefill waits.
        let i = mixed(
            &ids,
            &is_ctx,
            &plen,
            &done,
            &gen,
            &maxnew,
            &arrival,
            1,
            usize::MAX,
            0,
        );
        let (fitting, paused) = decide_indices(&i);
        assert_eq!(fitting.len(), 1, "only the decode that fits");
        assert!(paused.contains(&2), "the prefill must wait: {paused:?}");

        // Cap 2: nothing stalls, so the prefill starts.
        let i = mixed(
            &ids,
            &is_ctx,
            &plen,
            &done,
            &gen,
            &maxnew,
            &arrival,
            2,
            usize::MAX,
            0,
        );
        let (fitting, paused) = decide_indices(&i);
        assert!(
            fitting.contains(&2),
            "nothing stalled, so admit: {fitting:?}"
        );
        assert!(paused.is_empty());
    }

    /// Disaggregation makes the rule inert rather than harmful: a decode
    /// worker sees no context requests, so a permanently binding cap cannot
    /// starve a prefill that was never going to be scheduled here.
    #[test]
    fn the_rule_cannot_starve_prefill_on_a_disaggregated_worker() {
        let ids = [1i64, 2, 3];
        let is_ctx = [false, false, false];
        let plen = [4000u32, 4000, 4000];
        let done = [4000u32, 4000, 4000];
        let gen = [6u32, 6, 6];
        let maxnew = [200u32, 200, 200];
        let arrival = [0.0f64, 0.0, 0.0];
        let i = mixed(
            &ids,
            &is_ctx,
            &plen,
            &done,
            &gen,
            &maxnew,
            &arrival,
            1,
            usize::MAX,
            0,
        );
        let (fitting, paused) = decide_indices(&i);
        assert_eq!(fitting.len(), 1);
        assert_eq!(paused.len(), 2, "the other two decodes, and nothing else");
    }

    /// The watermark protects work in progress from work that has not begun,
    /// which is why it applies to a fresh prompt and not to a chunked prefill
    /// already underway.
    #[test]
    fn the_watermark_holds_back_new_work_and_not_work_in_progress() {
        let ids = [1i64, 2];
        let is_ctx = [true, true];
        let plen = [4000u32, 4000];
        let done = [0u32, 2000]; // one fresh, one half-computed
        let gen = [0u32, 0];
        let maxnew = [200u32, 200];
        let arrival = [0.0f64, 0.0];

        // Below the watermark.
        let i = mixed(
            &ids, &is_ctx, &plen, &done, &gen, &maxnew, &arrival, 8, 10, 100,
        );
        let (fitting, paused) = decide_indices(&i);
        assert_eq!(paused, vec![0], "the fresh prompt waits");
        assert_eq!(fitting, vec![1], "the one already underway continues");

        // Above it, both go.
        let i = mixed(
            &ids, &is_ctx, &plen, &done, &gen, &maxnew, &arrival, 8, 1000, 100,
        );
        let (fitting, _) = decide_indices(&i);
        assert_eq!(fitting.len(), 2);
    }
}
