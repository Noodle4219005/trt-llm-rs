use serde::Serialize;
use trtllm_core::{GoodputReport, LatencyStats};

/// Everything the simulation observed that is not the score itself.
///
/// These exist so that a goodput number can be *explained*. A run that scores
/// badly is either not finishing requests or not finishing them on time, and
/// the diagnostics say which stage is responsible - which is the thing that a
/// single number never tells you and a real run charges 60 SU per GPU-hour to
/// find out.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Diagnostics {
    pub simulated_s: f64,
    pub requests_issued: usize,
    pub requests_completed: usize,
    /// Mean sequences per prefill forward pass. Higher is more MoE-efficient
    /// and less deadline-fair; the scheduler trades between them on purpose.
    pub mean_prefill_batch_seqs: f64,
    pub mean_prefill_batch_tokens: f64,
    /// Fraction of prefill batches that stopped growing because of a deadline.
    /// Zero means the deadline rule never bound and the queue always had slack.
    pub deadline_limited_frac: f64,
    /// Fraction of requests the prefill scheduler gave up on.
    pub demoted_frac: f64,
    pub prefill_busy_frac: f64,
    pub mean_decode_concurrency: f64,
    pub peak_decode_concurrency: usize,
    pub final_decode_cap: f64,
    pub observed_step_ms: f64,
    pub decode_refusals: u64,
    pub prefill_queue_depth: LatencyStats,
    /// Decode concurrency the cost model was actually fitted at.
    pub calibrated_concurrency: f64,
    /// True when the run drove decode past the only concurrency anyone has
    /// measured on this model. The number is then an *extrapolation of an
    /// unidentified curve*, not a prediction, and must not be quoted as one.
    pub extrapolated_beyond_calibration: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct SimReport {
    pub goodput: GoodputReport,
    pub diagnostics: Diagnostics,
}

impl SimReport {
    /// One line suitable for a sweep table.
    pub fn summary(&self) -> String {
        let g = &self.goodput;
        format!(
            "goodput {:6.2} req/s | good {:5.1}% | req/s {:6.2} | TTFT p99 {:7.1} ms | ITL mean {:5.2} ms | decode C {:5.1} | batch {:.2} seq",
            g.goodput_req_s,
            g.good_frac * 100.0,
            g.req_per_s,
            g.ttft.p99,
            g.itl.mean,
            self.diagnostics.mean_decode_concurrency,
            self.diagnostics.mean_prefill_batch_seqs,
        )
    }

    /// Non-empty when the result depends on modelling nobody has measured.
    /// Print it next to the number, every time.
    pub fn caveats(&self) -> Vec<String> {
        let mut v = Vec::new();
        if self.diagnostics.extrapolated_beyond_calibration {
            v.push(format!(
                "decode ran at up to {} concurrent sequences but the ITL curve is calibrated at {:.0}; \
                 the goodput above is an extrapolation of an unidentified curve, not a prediction",
                self.diagnostics.peak_decode_concurrency, self.diagnostics.calibrated_concurrency
            ));
        }
        if self.goodput.total_requests < 100 {
            v.push(format!(
                "only {} requests fell inside the scored window; widen benchmark_s before comparing runs",
                self.goodput.total_requests
            ));
        }
        v
    }
}
