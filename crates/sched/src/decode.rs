//! Decode admission control.
//!
//! Decode sets the ceiling on `req/s`, and the ceiling is
//! `concurrency / (osl * itl)`. There are only two ways to raise it: run more
//! sequences at once, or spend more of the ITL budget per token. The measured
//! reference point does neither - 53 sequences at a mean ITL of 17.23 ms leaves
//! 14 % of a 20 ms budget unspent, which is 14 % of the score left on the table.
//!
//! Taking that headroom needs care, because the SLO is a *mean over the life of
//! one request*. Two consequences drive the design:
//!
//! * A request that has already emitted 150 tokens at 15 ms can absorb a much
//!   worse tail than one that has emitted 5. [`RunningSeq::tolerable_itl_ms`]
//!   makes that explicit, so the pool can run hot while its sequences are young
//!   and back off before anyone's *average* is spoiled.
//! * The ITL-versus-concurrency curve for this model is not known. One
//!   saturated point and one unsaturated sweep do not determine it, and fitting
//!   a line through them predicts concurrencies nobody has observed. So the cap
//!   is not read off a model at all: [`ItlController`] moves it by AIMD against
//!   measured step latency, the same way congestion control handles a link
//!   whose capacity it cannot see.

use std::collections::{HashMap, HashSet};

use trtllm_core::{Millis, RequestId};

/// A sequence currently decoding.
#[derive(Clone, Copy, Debug)]
pub struct RunningSeq {
    pub id: RequestId,
    /// When the decode scheduler took responsibility for this sequence.
    ///
    /// Distinct from `first_token_ms`, and the distinction is load-bearing: in
    /// disaggregated serving the first token is sampled by the PREFILL worker
    /// and the KV handoff happens after it, so `now - first_token_ms` on a
    /// freshly admitted sequence is 171 ms of transfer at the calibrated
    /// bandwidth. Steering on that makes every admission look like a stall the
    /// decode scheduler caused, and it throttles for a latency admission
    /// cannot fix.
    ///
    /// Set by `admit`. Zero until then.
    pub admitted_ms: Millis,
    pub first_token_ms: Millis,
    pub last_token_ms: Millis,
    pub tokens_emitted: u32,
    pub requested_tokens: u32,
}

impl RunningSeq {
    pub fn new(id: RequestId, first_token_ms: Millis, requested_tokens: u32) -> Self {
        Self {
            id,
            admitted_ms: first_token_ms,
            first_token_ms,
            last_token_ms: first_token_ms,
            tokens_emitted: 1,
            requested_tokens,
        }
    }

    pub fn remaining_tokens(&self) -> u32 {
        self.requested_tokens.saturating_sub(self.tokens_emitted)
    }

    pub fn is_done(&self) -> bool {
        self.tokens_emitted >= self.requested_tokens
    }

    /// Milliseconds already spent between the first and the most recent token.
    pub fn elapsed_ms(&self) -> f64 {
        self.last_token_ms - self.first_token_ms
    }

    /// Mean ITL over the tokens emitted so far.
    pub fn mean_itl_so_far_ms(&self) -> f64 {
        if self.tokens_emitted <= 1 {
            0.0
        } else {
            self.elapsed_ms() / f64::from(self.tokens_emitted - 1)
        }
    }

    /// The largest per-token latency this sequence can sustain for all of its
    /// *remaining* tokens and still finish with a mean ITL inside `budget_ms`.
    ///
    /// `(budget * gaps_total - elapsed) / gaps_remaining`, where
    /// `gaps_total = requested - 1`. Returns `f64::INFINITY` when there is
    /// nothing left to emit, and a negative number when the request is already
    /// unsalvageable - which is useful information, not an error.
    pub fn tolerable_itl_ms(&self, budget_ms: f64) -> f64 {
        let remaining = self.remaining_tokens();
        if remaining == 0 {
            return f64::INFINITY;
        }
        let gaps_total = f64::from(self.requested_tokens.saturating_sub(1));
        (budget_ms * gaps_total - self.elapsed_ms()) / f64::from(remaining)
    }
}

/// AIMD controller over the decode concurrency cap.
#[derive(Clone, Debug)]
pub struct ItlController {
    target_ms: f64,
    ewma_ms: f64,
    alpha: f64,
    cap: f64,
    min_cap: f64,
    max_cap: f64,
    increase_step: f64,
    decrease_factor: f64,
    low_water: f64,
    high_water: f64,
    /// Fraction of the cap the batch must actually reach before the cap is
    /// allowed to grow.
    utilisation_gate: f64,
    samples: u64,
    /// Set once lowering the cap has demonstrably stopped reducing ITL.
    ///
    /// The controller's model is that ITL rises with decode concurrency. That
    /// model is false whenever something else dominates the iteration -- a
    /// 4000-token prefill chunk stalls every generating sequence in the same
    /// iteration regardless of how few there are, and a model whose
    /// single-sequence step already exceeds the budget can never reach it by
    /// shedding load. In that regime backing off buys no latency and costs all
    /// the throughput, which is strictly worse than not backing off at all.
    ///
    /// Job 314882 is what this field is for: on Qwen3-235B at ISL 4000 the cap
    /// collapsed to min_cap = 1, refused 17k-48k admissions per worker, and
    /// still observed ~92 ms against a 20 ms target. Goodput was 0.00.
    concurrency_not_binding: bool,
    ewma_at_decrease: f64,
    observations_since_decrease: u64,
    /// How many observations to wait after a decrease before judging whether it
    /// helped. Long enough for the EWMA to actually move at alpha = 0.1.
    probe_window: u64,
    /// The decrease must buy at least this fraction of improvement to count.
    improvement_threshold: f64,
}

impl ItlController {
    pub fn new(target_ms: f64, initial_cap: f64, min_cap: f64, max_cap: f64) -> Self {
        Self {
            target_ms,
            ewma_ms: 0.0,
            alpha: 0.1,
            cap: initial_cap.clamp(min_cap, max_cap),
            min_cap,
            max_cap,
            increase_step: 1.0,
            decrease_factor: 0.9,
            low_water: 0.90,
            high_water: 0.98,
            utilisation_gate: 0.9,
            samples: 0,
            concurrency_not_binding: false,
            ewma_at_decrease: f64::INFINITY,
            observations_since_decrease: 0,
            probe_window: 15,
            improvement_threshold: 0.02,
        }
    }

    /// True once the controller has established that decode concurrency is not
    /// what is binding ITL. Reported so a run cannot quietly be interpreted as
    /// "the policy chose this": it means the policy gave up steering.
    pub fn concurrency_not_binding(&self) -> bool {
        self.concurrency_not_binding
    }

    pub fn target_ms(&self) -> f64 {
        self.target_ms
    }

    pub fn cap(&self) -> f64 {
        self.cap
    }

    pub fn observed_itl_ms(&self) -> f64 {
        self.ewma_ms
    }

    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// Feed one measured decode step. `step_ms` is the wall time of a forward
    /// pass, which is the inter-token latency every sequence in the batch sees,
    /// and `concurrency` is how many sequences were in it.
    ///
    /// `concurrency` is not decoration. A cap that is not the binding
    /// constraint must not grow: under a closed-loop client the batch size is
    /// set by arrivals, not by the cap, so a cap that keeps integrating upward
    /// while load is light reaches its ceiling and then needs a long run of
    /// multiplicative decreases to come back when load finally arrives. Same
    /// reason TCP does not open the congestion window while it is
    /// application-limited. The first end-to-end simulation caught exactly
    /// this: the cap finished at its 4096 ceiling with a real batch of 58.
    /// Feed one observation of the latency the SLO is written in.
    ///
    /// `itl_ms` must be a per-REQUEST inter-token latency across every request
    /// that has started generating, including ones producing nothing right now.
    /// Two cheaper quantities have been fed here and both were wrong, in the
    /// same direction, for the same reason -- they described the requests being
    /// served and were silent about the ones waiting:
    ///
    ///   - the scheduler iteration gap (job 314929: proxy read 11.71 ms while
    ///     AIPerf measured 39.10 ms on the same run), and
    ///   - the mean interval of sequences that advanced this step (job 316849:
    ///     read 15.4 ms across 26 advancing sequences while AIPerf measured
    ///     91 ms across 74 in decode; the controller never throttled and
    ///     goodput was 0.00).
    ///
    /// Both are equal to ITL only while every resident request advances every
    /// iteration. That is exactly the condition that stops holding when the
    /// engine starts rotating requests -- which is the situation this
    /// controller exists to detect.
    pub fn observe(&mut self, itl_ms: f64, concurrency: usize) {
        self.samples += 1;
        self.ewma_ms = if self.samples == 1 {
            itl_ms
        } else {
            (1.0 - self.alpha) * self.ewma_ms + self.alpha * itl_ms
        };

        // Do not steer on a handful of samples; a cold batch is not the steady
        // state and reacting to it oscillates.
        if self.samples < 8 {
            return;
        }
        // Budget reachable again: whatever was dominating the iteration has
        // gone, so resume steering. Requires the EWMA to be under the low water
        // mark, which at alpha = 0.1 already needs a sustained change rather
        // than one lucky iteration.
        if self.concurrency_not_binding && self.ewma_ms < self.target_ms * self.low_water {
            self.concurrency_not_binding = false;
            self.ewma_at_decrease = f64::INFINITY;
            self.observations_since_decrease = 0;
        }

        if self.ewma_ms > self.target_ms * self.high_water {
            if self.concurrency_not_binding {
                // Backing off has been shown not to help. Hold the cap: giving
                // up more throughput cannot buy latency that concurrency does
                // not control.
                return;
            }
            // Judging happens on a window; backing off still happens on every
            // observation. Slowing the back-off itself would trade one failure
            // mode for another -- a controller that reacts 15x slower to a
            // genuine overload is not an improvement on one that over-reacts.
            if self.ewma_at_decrease.is_infinite() {
                self.ewma_at_decrease = self.ewma_ms;
                self.observations_since_decrease = 0;
            }
            self.observations_since_decrease += 1;
            if self.observations_since_decrease >= self.probe_window {
                let improved =
                    self.ewma_ms <= self.ewma_at_decrease * (1.0 - self.improvement_threshold);
                if !improved {
                    // A whole window of decreases bought less than
                    // `improvement_threshold`. Concurrency is not the lever.
                    self.give_up();
                    return;
                }
                self.ewma_at_decrease = self.ewma_ms;
                self.observations_since_decrease = 0;
            }
            self.cap = (self.cap * self.decrease_factor).max(self.min_cap);

            // The floor, still over target. This is the decisive condition and
            // the improvement heuristic above does not catch it: while the cap
            // falls, ITL usually does drift down a little, so every window looks
            // like progress and the controller walks all the way to min_cap
            // without ever reaching the target. Job 314910 did exactly that --
            // cap 1.0 against a 20 ms target with observed ITL 95 ms, having
            // refused 22,073 admissions to buy nothing.
            //
            // One sequence at a time is as low as concurrency goes. If the
            // target is missed there, it is not concurrency that is missing it.
            if self.cap <= self.min_cap * 1.01 {
                self.give_up();
            }
        } else if self.ewma_ms < self.target_ms * self.low_water
            && concurrency as f64 >= self.cap * self.utilisation_gate
        {
            self.cap = (self.cap + self.increase_step).min(self.max_cap);
            self.ewma_at_decrease = f64::INFINITY;
            self.observations_since_decrease = 0;
        }
    }

    /// Stop steering, and undo the throttling that bought nothing.
    ///
    /// Restoring the cap is the point, not a side effect. A controller that
    /// merely stopped decreasing would sit at whatever cap it had already
    /// collapsed to, which is the state that produced goodput 0.00. If
    /// concurrency does not move latency, then holding concurrency down has no
    /// benefit to trade against its cost, so the cap goes back to where
    /// throughput is best and stays there until the budget is reachable again.
    fn give_up(&mut self) {
        self.concurrency_not_binding = true;
        self.cap = self.max_cap;
        self.ewma_at_decrease = f64::INFINITY;
        self.observations_since_decrease = 0;
    }

    /// Force the cap down, e.g. because an in-flight request is about to blow
    /// its own average.
    pub fn back_off(&mut self) {
        self.cap = (self.cap * self.decrease_factor).max(self.min_cap);
    }
}

/// Why the scheduler did or did not take a new sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmitDecision {
    Admit,
    /// The concurrency cap is already reached.
    AtCap,
    /// Admitting would push an in-flight request past its own ITL average.
    WouldSpoilRunning,
    /// The worker has no KV headroom.
    NoKvHeadroom,
}

impl AdmitDecision {
    pub fn is_admit(self) -> bool {
        matches!(self, AdmitDecision::Admit)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            AdmitDecision::Admit => "admit",
            AdmitDecision::AtCap => "at-cap",
            AdmitDecision::WouldSpoilRunning => "would-spoil-running",
            AdmitDecision::NoKvHeadroom => "no-kv-headroom",
        }
    }
}

#[derive(Debug)]
pub struct DecodeScheduler {
    running: HashMap<RequestId, RunningSeq>,
    controller: ItlController,
    itl_budget_ms: f64,
    /// How much of the measured step latency an in-flight request must still be
    /// able to absorb before another sequence is let in. 1.0 is break-even;
    /// above 1.0 keeps a margin against the step getting slower.
    risk_margin: f64,
    /// Tokens one accepted step emits per sequence. 1 without speculation,
    /// 1 + draft_tokens with it.
    ///
    /// This is the ceiling, not the expectation: EAGLE3 at one draft token
    /// emits two on acceptance and one on rejection, and the acceptance rate
    /// (1.82 measured elsewhere) is not known on this stack. Overstating it
    /// would make `remaining_tokens` finish sequences early. The engine tells
    /// us which sequences advanced, not by how much, so the honest options are
    /// this ceiling or a per-request delta from the bridge -- and the bridge
    /// already reports `tokens_generated`, which is the delta's source when
    /// someone wires it.
    tokens_per_step: u32,
    admitted: u64,
    refused: u64,
    /// Sequences the engine did not report as finished, even though our own
    /// bookkeeping says they have emitted their full token budget. Counted,
    /// never acted on: see [`DecodeScheduler::on_step`].
    ///
    /// Counted once per *sequence*, not once per step. A sequence in this
    /// state stays running, so a per-step count would grow without bound for
    /// as long as it lived and would say more about how long we watched than
    /// about how often the two sides disagree.
    pub finish_disagreements: u64,
    /// Ids already counted in `finish_disagreements`, so each is counted once.
    /// Bounded by concurrency: entries leave when the sequence is retired.
    flagged: HashSet<RequestId>,
    /// `finished` ids the engine reported for a request we were not tracking
    /// as running (already removed, or never admitted).
    pub unknown_finish_reports: u64,
}

impl DecodeScheduler {
    pub fn new(itl_budget_ms: f64, controller: ItlController) -> Self {
        Self {
            running: HashMap::new(),
            controller,
            itl_budget_ms,
            risk_margin: 1.05,
            tokens_per_step: 1,
            admitted: 0,
            refused: 0,
            finish_disagreements: 0,
            flagged: HashSet::new(),
            unknown_finish_reports: 0,
        }
    }

    pub fn controller(&self) -> &ItlController {
        &self.controller
    }

    pub fn concurrency(&self) -> usize {
        self.running.len()
    }

    pub fn admitted(&self) -> u64 {
        self.admitted
    }

    pub fn refused(&self) -> u64 {
        self.refused
    }

    pub fn running(&self) -> impl Iterator<Item = &RunningSeq> {
        self.running.values()
    }

    /// The tightest ITL tolerance among the sequences in flight. This is the
    /// number that says how much room the pool actually has, and it is not the
    /// same as "budget minus current ITL": a request that has been slow so far
    /// is already spending its future.
    pub fn tightest_tolerance_ms(&self) -> f64 {
        self.running
            .values()
            .map(|s| s.tolerable_itl_ms(self.itl_budget_ms))
            .fold(f64::INFINITY, f64::min)
    }

    /// Can one more sequence start decoding right now?
    pub fn can_admit(&self, kv_headroom: bool) -> AdmitDecision {
        if !kv_headroom {
            return AdmitDecision::NoKvHeadroom;
        }
        if (self.running.len() as f64) >= self.controller.cap() {
            return AdmitDecision::AtCap;
        }
        // Only meaningful once the controller has an opinion about step latency.
        if self.controller.samples() >= 8 {
            let tightest = self.tightest_tolerance_ms();
            if tightest.is_finite()
                && tightest < self.controller.observed_itl_ms() * self.risk_margin
            {
                return AdmitDecision::WouldSpoilRunning;
            }
        }
        AdmitDecision::Admit
    }

    /// Record a decision so the refusal reasons show up in metrics whether or
    /// not the caller went ahead.
    pub fn note(&mut self, decision: AdmitDecision) {
        if decision.is_admit() {
            self.admitted += 1;
        } else {
            self.refused += 1;
            if decision == AdmitDecision::WouldSpoilRunning {
                self.controller.back_off();
            }
        }
    }

    /// Take responsibility for a sequence as of `now`.
    pub fn admit_at(&mut self, mut seq: RunningSeq, now: Millis) {
        seq.admitted_ms = now;
        seq.last_token_ms = seq.last_token_ms.max(now);
        self.admit(seq)
    }

    pub fn admit(&mut self, seq: RunningSeq) {
        self.running.insert(seq.id, seq);
    }

    /// Apply the engine's report of what happened during one decode step.
    ///
    /// The engine is the sole authority on which requests advanced and which
    /// finished - this scheduler no longer infers either. That contract
    /// ("one step = one token for every running sequence, and we can tell for
    /// ourselves when a sequence is done") was refuted by job 312007: with
    /// TensorRT-LLM's overlap scheduler on, the engine measured 3.19 ms ITL /
    /// 64.88 req/s; forced off, 5.34 ms ITL / 42.47 req/s (+67.3% ITL, -34.5%
    /// throughput), so overlap must stay on. With overlap on, `decode_step`
    /// returns the *previous* step's tokens, and not every running sequence
    /// is guaranteed to advance in a given step.
    ///
    /// An earlier version of this comment argued that a pipelined step's wall
    /// time is still the inter-token latency the controller needs to see. Job
    /// 316849 refuted it: the engine advanced 26 sequences every 15.4 ms while
    /// holding 74 in decode, so the step time was inside the 20 ms budget and
    /// the latency AIPerf scored was 91 ms. Step time equals ITL only while
    /// every resident request advances every step, and a controller that
    /// assumes it is blind in precisely the regime it exists for.
    /// Tell the scheduler how many tokens one accepted step emits.
    ///
    /// `1 + draft_tokens` under speculation. Set it from the same place that
    /// sets `speculative_config`, or the two disagree and the scheduler steers
    /// against a token count the engine is not producing.
    pub fn set_tokens_per_step(&mut self, tokens: u32) {
        self.tokens_per_step = tokens.max(1);
    }

    pub fn tokens_per_step(&self) -> u32 {
        self.tokens_per_step
    }

    pub fn on_step(
        &mut self,
        now: Millis,
        step_ms: f64,
        advanced: &[RequestId],
        finished: &[RequestId],
    ) -> Vec<RunningSeq> {
        // Steer BEFORE booking this step's tokens. Measured after, every
        // advancing sequence reads an age of zero and the signal collapses to
        // 0.00 ms -- which is what `a_healthy_batch_reports_the_step_time`
        // caught. Measured before, a sequence that is about to advance still
        // shows the interval it just completed, and one that is not shows how
        // long it has been starving. Both are the quantity we want.
        //
        // `step_ms` is the fallback only when nothing is running, because with
        // no request there is no per-request latency to observe.
        let steer = self.steering_itl_ms(now).unwrap_or(step_ms);
        self.controller.observe(steer, self.running.len());

        // One appearance is not one token under speculation: a verified step
        // emits 1 + accepted drafts. Booking one each would understate
        // `tokens_emitted`, which `remaining_tokens` and `tolerable_itl_ms`
        // both divide by, so a speculating deployment would think every
        // sequence was further behind than it is and steer against a fiction.
        //
        // ADR 0036 recorded this as a latent defect that could not be verified
        // while the SU budget was withdrawn. Enabling SPECULATION made it live.
        let per_step = self.tokens_per_step.max(1);
        for id in advanced {
            if let Some(seq) = self.running.get_mut(id) {
                seq.tokens_emitted = seq
                    .tokens_emitted
                    .saturating_add(per_step)
                    .min(seq.requested_tokens);
                seq.last_token_ms = now;
            }
        }

        let mut done = Vec::new();
        for id in finished {
            match self.running.remove(id) {
                Some(seq) => {
                    self.flagged.remove(id);
                    done.push(seq);
                }
                None => self.unknown_finish_reports += 1,
            }
        }

        // Our count says a sequence has emitted its whole budget but the engine
        // did not list it in `finished`. This is NOT expected pipeline skew:
        // `tokens` and `finished` arrive in the same `DecodeStepOutcome` and so
        // describe the same engine step. It means the two sides are keeping
        // different books - most likely our `requested_tokens` does not match
        // what the engine was actually told. The engine still wins, so the
        // sequence is left running; the counter exists so a persistent mismatch
        // is visible instead of silently capping every request one token early.
        for seq in self.running.values() {
            if seq.is_done() && self.flagged.insert(seq.id) {
                self.finish_disagreements += 1;
            }
        }

        done
    }

    /// p90 of how long each running request has been waiting for its next
    /// token.
    ///
    /// Equal to the step time while every resident request advances every step,
    /// and growing exactly when the engine starts starving some of them --
    /// which is the regime a step-time signal is blind to and this controller
    /// exists for.
    ///
    /// Time since the LAST token, not an average since the first. Averaging
    /// from `first_token_ms` was tried and the simulator rejected it: that
    /// anchor is when prefill emitted the first token, which can precede
    /// decode admission by a long handoff, so every freshly admitted request
    /// arrived carrying a large "ITL" that decode concurrency cannot fix.
    /// `goodput_saturates_and_then_ttft_degrades` collapsed to zero completed
    /// requests. Starvation age carries no history from before admission.
    ///
    /// p90 rather than the mean because the pass criterion is
    /// `good_frac >= 0.90`: the ninetieth percentile is the request the score
    /// actually turns on.
    fn steering_itl_ms(&self, now: Millis) -> Option<f64> {
        // Age since the LATER of the last token and admission.
        //
        // Excluding fresh sequences outright was tried and was wrong twice: it
        // drops the starving request the controller exists to notice (its
        // token count is 1 and never grows), and it did not move the
        // simulator's goodput. Clamping at admission keeps every sequence in
        // the sample while charging none of them for the handoff that happened
        // before the decode scheduler could do anything about it.
        let mut samples: Vec<f64> = self
            .running
            .values()
            .map(|seq| now - seq.last_token_ms.max(seq.admitted_ms))
            .collect();
        if samples.is_empty() {
            return None;
        }
        samples.sort_by(|a, b| a.partial_cmp(b).expect("itl samples are finite"));
        let idx = (((samples.len() - 1) as f64) * 0.90).round() as usize;
        Some(samples[idx])
    }

    /// Simulator- and unit-test-only convenience: builds `advanced` from
    /// every running id and `finished` from every running id whose next
    /// token would reach its budget, then delegates to [`Self::on_step`].
    /// This reproduces the scheduler's old, pre-job-312007 behaviour, which
    /// is the correct model *for the simulator* because the simulator's curve
    /// model advances every sequence every step by construction.
    ///
    /// A real deployment must not use this: it must pass the engine's own
    /// `advanced`/`finished` lists to `on_step`, because "every sequence
    /// advances every step" is a simulator modelling choice, not a fact about
    /// the runtime.
    pub fn on_step_synthetic(&mut self, now: Millis, step_ms: f64) -> Vec<RunningSeq> {
        let advanced: Vec<RequestId> = self.running.keys().copied().collect();
        let finished: Vec<RequestId> = self
            .running
            .values()
            .filter(|s| s.tokens_emitted + 1 >= s.requested_tokens)
            .map(|s| s.id)
            .collect();
        self.on_step(now, step_ms, &advanced, &finished)
    }

    pub fn remove(&mut self, id: RequestId) -> Option<RunningSeq> {
        self.flagged.remove(&id);
        self.running.remove(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One appearance is not one token under speculation.
    ///
    /// ADR 0036 recorded this as a latent defect: `remaining_tokens` and
    /// `tolerable_itl_ms` both divide by `tokens_emitted`, so booking one token
    /// per advancing sequence while the engine emits two would make every
    /// sequence look further behind than it is and the controller steer against
    /// a fiction. Enabling SPECULATION is what made it live.
    #[test]
    fn a_speculating_step_books_more_than_one_token() {
        let mk = || DecodeScheduler::new(20.0, ItlController::new(20.0, 8.0, 1.0, 8.0));
        let mut plain = mk();
        let mut spec = mk();
        spec.set_tokens_per_step(2); // EAGLE3 at one draft token

        plain.admit_at(RunningSeq::new(RequestId(1), 0.0, 200), 0.0);
        spec.admit_at(RunningSeq::new(RequestId(1), 0.0, 200), 0.0);
        for step in 1..=10 {
            let now = f64::from(step) * 12.0;
            plain.on_step(now, 12.0, &[RequestId(1)], &[]);
            spec.on_step(now, 12.0, &[RequestId(1)], &[]);
        }

        let p = *plain.running().next().expect("plain seq");
        let q = *spec.running().next().expect("spec seq");
        assert_eq!(p.tokens_emitted, 11, "1 at admission plus ten steps");
        assert_eq!(q.tokens_emitted, 21, "1 at admission plus ten double steps");
        assert!(
            q.remaining_tokens() < p.remaining_tokens(),
            "the speculating sequence must be further ahead, not behind"
        );
    }

    /// The count must not run past what was asked for, or a sequence would be
    /// reported as owing negative work.
    #[test]
    fn speculation_cannot_overshoot_the_requested_length() {
        let mut s = DecodeScheduler::new(20.0, ItlController::new(20.0, 8.0, 1.0, 8.0));
        s.set_tokens_per_step(4);
        s.admit_at(RunningSeq::new(RequestId(1), 0.0, 10), 0.0);
        for step in 1..=10 {
            s.on_step(f64::from(step) * 12.0, 12.0, &[RequestId(1)], &[]);
        }
        let seq = s.running().next().copied();
        if let Some(seq) = seq {
            assert!(seq.tokens_emitted <= 10, "{}", seq.tokens_emitted);
            assert_eq!(seq.remaining_tokens(), 0);
        }
    }

    /// Zero would stall every sequence for ever; the setter must refuse it.
    #[test]
    fn tokens_per_step_is_at_least_one() {
        let mut s = DecodeScheduler::new(20.0, ItlController::new(20.0, 8.0, 1.0, 8.0));
        s.set_tokens_per_step(0);
        assert_eq!(s.tokens_per_step(), 1);
    }

    fn seq_at(emitted: u32, elapsed: f64, requested: u32) -> RunningSeq {
        RunningSeq {
            id: RequestId(0),
            admitted_ms: 0.0,
            first_token_ms: 0.0,
            last_token_ms: elapsed,
            tokens_emitted: emitted,
            requested_tokens: requested,
        }
    }

    /// A young request can absorb almost the whole budget; an old slow one
    /// cannot. Treating them the same is what makes a naive batch-size cap
    /// either too timid or too late.
    #[test]
    fn tolerance_depends_on_how_much_of_the_average_is_already_spent() {
        // Fresh: one token out, 199 gaps to go, 20 ms budget.
        let fresh = seq_at(1, 0.0, 200);
        assert!((fresh.tolerable_itl_ms(20.0) - 20.0).abs() < 0.01);

        // Ran cheap for 150 tokens at 15 ms: it has banked a lot of slack.
        let banked = seq_at(150, 149.0 * 15.0, 200);
        assert!(
            banked.tolerable_itl_ms(20.0) > 33.0,
            "{}",
            banked.tolerable_itl_ms(20.0)
        );

        // Ran expensive for 150 tokens at 21 ms: already over, no room left.
        let spent = seq_at(150, 149.0 * 21.0, 200);
        assert!(
            spent.tolerable_itl_ms(20.0) < 18.0,
            "{}",
            spent.tolerable_itl_ms(20.0)
        );
    }

    #[test]
    fn a_finished_sequence_tolerates_anything() {
        assert!(seq_at(200, 4000.0, 200)
            .tolerable_itl_ms(20.0)
            .is_infinite());
    }

    #[test]
    fn controller_raises_the_cap_while_latency_is_under_budget() {
        let mut c = ItlController::new(20.0, 40.0, 8.0, 256.0);
        for _ in 0..40 {
            let at_cap = c.cap() as usize;
            c.observe(15.0, at_cap);
        }
        assert!(
            c.cap() > 40.0,
            "cap should climb into the unused budget: {}",
            c.cap()
        );
        assert!(c.cap() <= 256.0);
    }

    /// The cap must stay put when the batch is nowhere near it. Otherwise it
    /// integrates to its ceiling under light load and is useless as a brake the
    /// moment load arrives.
    #[test]
    fn an_unbinding_cap_does_not_grow() {
        let mut c = ItlController::new(20.0, 64.0, 8.0, 4096.0);
        for _ in 0..200 {
            c.observe(12.0, 4);
        }
        assert!((c.cap() - 64.0).abs() < 1e-9, "cap drifted to {}", c.cap());
    }

    #[test]
    fn controller_backs_off_when_latency_crosses_the_target() {
        let mut c = ItlController::new(20.0, 64.0, 8.0, 256.0);
        for _ in 0..40 {
            let at_cap = c.cap() as usize;
            c.observe(15.0, at_cap);
        }
        let hot = c.cap();

        // Within one probe window the controller has not yet had the chance to
        // learn whether backing off helps, so it backs off -- which is the
        // behaviour this test has always been about.
        for _ in 0..10 {
            c.observe(26.0, 64);
        }
        assert!(
            c.cap() < hot,
            "cap must fall once ITL exceeds target: {hot} -> {}",
            c.cap()
        );
        assert!(c.cap() >= 8.0, "cap must not collapse below the floor");

        // Beyond that window the latency here has not moved at all, so the
        // controller is entitled to conclude that concurrency is not what is
        // holding ITL above target, and to stop paying for a lever that does
        // nothing. Feeding a constant latency is exactly the "not binding" case.
        for _ in 0..60 {
            c.observe(26.0, 64);
        }
        assert!(
            c.concurrency_not_binding(),
            "latency that ignores the cap means the cap is not the lever"
        );
    }

    #[test]
    fn the_cap_stops_falling_once_falling_stops_helping() {
        // Qwen3-235B at ISL 4000 (job 314882): every iteration carries a 4000
        // token prefill chunk that stalls each generating sequence, so ITL does
        // not move with decode concurrency at all. The old controller took that
        // to min_cap = 1, refused tens of thousands of admissions, and still
        // missed the target -- goodput 0.00.
        let mut c = ItlController::new(20.0, 64.0, 1.0, 256.0);
        for _ in 0..600 {
            c.observe(92.0, c.cap() as usize);
        }
        assert!(
            c.concurrency_not_binding(),
            "controller must notice that backing off is not buying latency"
        );
        assert!(
            c.cap() > 1.0,
            "cap must not collapse to the floor when the floor cannot meet the \
             target either: cap = {}",
            c.cap()
        );
    }

    #[test]
    fn a_cap_that_is_binding_is_still_reduced() {
        // The escape hatch must not disarm the controller when concurrency
        // really is what drives ITL. Here latency falls with the cap, so every
        // decrease pays for itself and the controller must keep going.
        let mut c = ItlController::new(20.0, 64.0, 1.0, 256.0);
        let start = c.cap();
        for _ in 0..600 {
            // ITL proportional to the cap: halving the batch halves latency.
            let itl = c.cap() * 0.5;
            c.observe(itl, c.cap() as usize);
        }
        assert!(
            !c.concurrency_not_binding(),
            "concurrency is binding here; the controller must not give up"
        );
        assert!(
            c.cap() < start,
            "cap must fall while falling helps: {start} -> {}",
            c.cap()
        );
        assert!(
            c.observed_itl_ms() <= 20.0,
            "the controller should have reached the target: {}",
            c.observed_itl_ms()
        );
    }

    #[test]
    fn giving_up_is_reversible() {
        // A workload phase change -- the long prefills drain -- must let the
        // controller steer again, or one bad phase disables it for the run.
        let mut c = ItlController::new(20.0, 64.0, 1.0, 256.0);
        for _ in 0..600 {
            c.observe(92.0, c.cap() as usize);
        }
        assert!(c.concurrency_not_binding());
        for _ in 0..200 {
            c.observe(5.0, c.cap() as usize);
        }
        assert!(
            !c.concurrency_not_binding(),
            "the flag must clear once the budget is reachable again"
        );
    }

    #[test]
    fn giving_up_restores_the_throughput_the_throttling_cost() {
        // Not merely "stops falling": the cap must come back. Job 314910 sat at
        // cap 1.0 having refused 22,073 admissions to buy nothing, and a fix
        // that only froze it there would have kept every bit of that damage.
        let mut c = ItlController::new(20.0, 128.0, 1.0, 128.0);
        for _ in 0..400 {
            // Latency that improves slightly as the cap falls -- enough for the
            // per-window improvement test to keep saying "progress" -- but never
            // reaches the target. This is the shape that walked the old
            // controller to the floor.
            let itl = 90.0 + c.cap() * 0.05;
            c.observe(itl, c.cap() as usize);
        }
        assert!(c.concurrency_not_binding());
        assert!(
            c.cap() >= 128.0 * 0.99,
            "the cap must be restored, not frozen at whatever it collapsed to: {}",
            c.cap()
        );
    }

    #[test]
    fn a_single_fast_step_must_not_rearm_the_collapse() {
        // Reproduces job 314910: the run is a mix of cheap chunked-prefill
        // iterations and expensive ones. One dip below target*low_water cleared
        // concurrency_not_binding, the decreases resumed, and the cap still ended
        // at the floor.
        let mut c = ItlController::new(20.0, 128.0, 1.0, 128.0);
        for i in 0..2000 {
            let step = if i % 25 == 0 { 5.0 } else { 95.0 };
            c.observe(step, c.cap() as usize);
        }
        assert!(
            c.cap() > 1.0,
            "an occasional fast iteration must not let the cap collapse: cap = {}",
            c.cap()
        );
    }

    #[test]
    fn admission_stops_at_the_cap() {
        let mut s = DecodeScheduler::new(20.0, ItlController::new(20.0, 2.0, 1.0, 8.0));
        s.admit(RunningSeq::new(RequestId(1), 0.0, 200));
        s.admit(RunningSeq::new(RequestId(2), 0.0, 200));
        assert_eq!(s.can_admit(true), AdmitDecision::AtCap);
    }

    #[test]
    fn no_kv_headroom_beats_every_other_reason() {
        let s = DecodeScheduler::new(20.0, ItlController::new(20.0, 64.0, 1.0, 256.0));
        assert_eq!(s.can_admit(false), AdmitDecision::NoKvHeadroom);
    }

    #[test]
    fn admission_is_refused_when_an_inflight_request_has_no_room_left() {
        let mut s = DecodeScheduler::new(20.0, ItlController::new(20.0, 64.0, 1.0, 256.0));
        // One request that has already burned its average.
        s.admit(RunningSeq {
            id: RequestId(7),
            admitted_ms: 0.0,
            first_token_ms: 0.0,
            last_token_ms: 149.0 * 22.0,
            tokens_emitted: 150,
            requested_tokens: 200,
        });
        for _ in 0..10 {
            s.controller.observe(19.0, 1);
        }
        assert_eq!(s.can_admit(true), AdmitDecision::WouldSpoilRunning);
    }

    /// The failure job 316849 shipped with: the engine advances a small subset
    /// every step while the rest sit resident and starving. The step time stays
    /// inside the budget, so a controller fed step time sees nothing; the
    /// latency the benchmark scores is several times worse. The signal must
    /// come from the requests that are NOT moving.
    #[test]
    fn a_starving_request_raises_the_signal_even_while_steps_are_fast() {
        let mut s = DecodeScheduler::new(20.0, ItlController::new(20.0, 64.0, 1.0, 256.0));
        // Two sequences the engine keeps advancing, one it has abandoned.
        s.admit(RunningSeq::new(RequestId(1), 0.0, 100));
        s.admit(RunningSeq::new(RequestId(2), 0.0, 100));
        s.admit(RunningSeq::new(RequestId(3), 0.0, 100));

        // Ten fast steps, but request 3 never appears in `advanced`.
        let mut now = 0.0;
        for _ in 0..10 {
            now += 15.0;
            s.on_step(now, 15.0, &[RequestId(1), RequestId(2)], &[]);
        }

        // Requests 1 and 2 are at 15 ms. Request 3 has gone 150 ms without a
        // second token, and it is the one the score turns on.
        let observed = s.controller.observed_itl_ms();
        assert!(
            observed > 20.0,
            "controller saw {observed:.1} ms; a request starving for 150 ms must \
             not read as inside a 20 ms budget"
        );
    }

    /// The converse, so the fix cannot be "always report something alarming":
    /// when every resident request advances every step, the signal is the step.
    #[test]
    fn a_healthy_batch_reports_the_step_time() {
        let mut s = DecodeScheduler::new(20.0, ItlController::new(20.0, 64.0, 1.0, 256.0));
        s.admit(RunningSeq::new(RequestId(1), 0.0, 100));
        s.admit(RunningSeq::new(RequestId(2), 0.0, 100));

        let mut now = 0.0;
        for _ in 0..10 {
            now += 15.0;
            s.on_step(now, 15.0, &[RequestId(1), RequestId(2)], &[]);
        }

        let observed = s.controller.observed_itl_ms();
        assert!(
            (observed - 15.0).abs() < 1.0,
            "expected ~15 ms, got {observed:.2} ms"
        );
    }

    #[test]
    fn stepping_retires_sequences_that_reached_their_token_count() {
        let mut s = DecodeScheduler::new(20.0, ItlController::new(20.0, 64.0, 1.0, 256.0));
        s.admit(RunningSeq::new(RequestId(1), 0.0, 3));
        assert!(s.on_step_synthetic(17.0, 17.0).is_empty());
        let done = s.on_step_synthetic(34.0, 17.0);
        assert_eq!(done.len(), 1);
        assert_eq!(s.concurrency(), 0);
        assert_eq!(done[0].tokens_emitted, 3);
    }

    /// The engine's `advanced` list is authoritative: a running sequence left
    /// out of it must not be credited with a token it was not reported to
    /// have received.
    #[test]
    fn a_sequence_not_in_advanced_is_not_incremented() {
        let mut s = DecodeScheduler::new(20.0, ItlController::new(20.0, 64.0, 1.0, 256.0));
        s.admit(RunningSeq::new(RequestId(1), 0.0, 10));
        let done = s.on_step(10.0, 10.0, &[], &[]);
        assert!(done.is_empty());
        let seq = s.running().find(|r| r.id == RequestId(1)).unwrap();
        assert_eq!(seq.tokens_emitted, 1);
    }

    /// The engine's `finished` list is authoritative in the other direction
    /// too: a request it names is retired even if our own `is_done()` would
    /// have said "not yet".
    #[test]
    fn a_sequence_in_finished_is_returned_even_when_not_done() {
        let mut s = DecodeScheduler::new(20.0, ItlController::new(20.0, 64.0, 1.0, 256.0));
        s.admit(RunningSeq::new(RequestId(1), 0.0, 200));
        let done = s.on_step(10.0, 10.0, &[RequestId(1)], &[RequestId(1)]);
        assert_eq!(done.len(), 1);
        assert_eq!(s.concurrency(), 0);
        assert!(!done[0].is_done());
    }

    /// `is_done()` looking true is not, by itself, grounds to retire a
    /// sequence: only the engine's `finished` list is. A disagreement is
    /// counted and the sequence is left running.
    #[test]
    fn is_done_without_an_engine_finish_report_stays_running() {
        let mut s = DecodeScheduler::new(20.0, ItlController::new(20.0, 64.0, 1.0, 256.0));
        // requested_tokens == 1: RunningSeq::new starts tokens_emitted at 1,
        // so this sequence is already is_done() before any step is taken.
        s.admit(RunningSeq::new(RequestId(1), 0.0, 1));
        let done = s.on_step(10.0, 10.0, &[], &[]);
        assert!(done.is_empty());
        assert_eq!(s.concurrency(), 1);
        assert_eq!(s.finish_disagreements, 1);

        // Counted once per sequence, not once per step. A per-step count would
        // measure how long we watched rather than how often the two sides
        // disagree, and one stuck request would run it away on its own.
        for _ in 0..5 {
            s.on_step(10.0, 10.0, &[], &[]);
        }
        assert_eq!(s.finish_disagreements, 1);
    }
}
