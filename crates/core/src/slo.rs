use serde::{Deserialize, Serialize};

/// The service-level objective a request is judged against.
///
/// Defaults are the 2026 APAC HPC-AI Qwen3 rules: a request is *good* when
/// `TTFT <= 3000 ms` **and** its **mean** inter-token latency is `<= 20 ms`.
/// The benchmark passes when at least 90 % of requests are good, and the score
/// is the output-token rate of the good requests only.
///
/// Two properties of that definition drive every policy in this repository:
///
/// 1. It is **per request**, not a percentile over the fleet. One request
///    blowing its TTFT costs exactly one request, so a scheduler is allowed to
///    sacrifice a request on purpose if that keeps two others on time.
/// 2. The ITL bound is a **mean over the whole life of the request**. Admitting
///    a sequence late therefore damages every sequence already decoding, which
///    is why the decode side needs an admission gate and not just a queue.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Slo {
    /// Time to first token budget, milliseconds.
    pub ttft_ms: f64,
    /// Mean inter-token latency budget, milliseconds.
    pub itl_ms: f64,
    /// Fraction of requests that must be good for the run to pass.
    pub good_frac_target: f64,
}

impl Default for Slo {
    fn default() -> Self {
        Self {
            ttft_ms: 3000.0,
            itl_ms: 20.0,
            good_frac_target: 0.90,
        }
    }
}

impl Slo {
    /// Seconds a good request occupies a decode slot: it must emit `osl`
    /// tokens and may not average more than `itl_ms` between them.
    ///
    /// This is the single most load-bearing number in the whole design. The
    /// decode side cannot serve more than `concurrency / slot_seconds`
    /// requests per second no matter how fast the kernels are.
    pub fn decode_slot_seconds(&self, osl: u32) -> f64 {
        f64::from(osl) * self.itl_ms / 1000.0
    }

    /// Upper bound on request rate for a decode pool that sustains
    /// `concurrency` simultaneous sequences.
    pub fn decode_req_per_s(&self, concurrency: f64, osl: u32) -> f64 {
        concurrency / self.decode_slot_seconds(osl)
    }
}

/// Why a request was or was not counted as good.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    Good,
    /// Missed the time-to-first-token budget (queueing on the prefill side).
    LateFirstToken,
    /// Mean inter-token latency exceeded the budget (decode oversubscribed).
    SlowTokens,
    /// Both budgets were missed.
    LateAndSlow,
    /// The request never produced the requested number of tokens.
    Incomplete,
}

impl Verdict {
    pub fn is_good(self) -> bool {
        matches!(self, Verdict::Good)
    }
}
