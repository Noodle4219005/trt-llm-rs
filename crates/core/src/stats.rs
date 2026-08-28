use serde::{Deserialize, Serialize};

use crate::request::RequestOutcome;
use crate::slo::{Slo, Verdict};

/// Percentile summary of a latency sample. Kept as an explicit sorted vector
/// rather than a histogram: a benchmark window holds a few thousand requests,
/// exact percentiles are cheap, and approximate ones have burned us before.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LatencyStats {
    pub count: usize,
    pub mean: f64,
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
}

impl LatencyStats {
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }
        let mut v = samples.to_vec();
        v.sort_by(f64::total_cmp);
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        Self {
            count: v.len(),
            mean,
            p50: percentile(&v, 0.50),
            p90: percentile(&v, 0.90),
            p95: percentile(&v, 0.95),
            p99: percentile(&v, 0.99),
            max: *v.last().expect("non-empty"),
        }
    }
}

/// Nearest-rank percentile of an already sorted slice.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (q * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// The scored result of a benchmark window.
///
/// `goodput_req_s` is the number the competition ranks on. Everything else is
/// here so that a regression can be attributed instead of merely noticed: a
/// drop in `goodput_req_s` is either fewer requests finished (`req_per_s`) or
/// more of them missing the SLO (`good_frac`), and the verdict breakdown says
/// which budget was missed.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GoodputReport {
    pub window_s: f64,
    pub total_requests: usize,
    pub good_requests: usize,
    pub good_frac: f64,
    pub req_per_s: f64,
    pub goodput_req_s: f64,
    /// Output tokens per second counting only good requests - the scored metric.
    pub good_output_tok_s: f64,
    pub late_first_token: usize,
    pub slow_tokens: usize,
    pub late_and_slow: usize,
    pub incomplete: usize,
    pub ttft: LatencyStats,
    pub itl: LatencyStats,
    pub passed: bool,
}

impl GoodputReport {
    pub fn from_outcomes(outcomes: &[RequestOutcome], window_s: f64, slo: &Slo) -> Self {
        let mut r = Self {
            window_s,
            total_requests: outcomes.len(),
            ..Default::default()
        };
        let mut ttfts = Vec::with_capacity(outcomes.len());
        let mut itls = Vec::with_capacity(outcomes.len());
        let mut good_tokens = 0u64;

        for o in outcomes {
            ttfts.push(o.ttft_ms());
            itls.push(o.mean_itl_ms());
            match o.verdict(slo) {
                Verdict::Good => {
                    r.good_requests += 1;
                    good_tokens += u64::from(o.output_tokens);
                }
                Verdict::LateFirstToken => r.late_first_token += 1,
                Verdict::SlowTokens => r.slow_tokens += 1,
                Verdict::LateAndSlow => r.late_and_slow += 1,
                Verdict::Incomplete => r.incomplete += 1,
            }
        }

        if r.total_requests > 0 {
            r.good_frac = r.good_requests as f64 / r.total_requests as f64;
        }
        if window_s > 0.0 {
            r.req_per_s = r.total_requests as f64 / window_s;
            r.goodput_req_s = r.good_requests as f64 / window_s;
            r.good_output_tok_s = good_tokens as f64 / window_s;
        }
        r.ttft = LatencyStats::from_samples(&ttfts);
        r.itl = LatencyStats::from_samples(&itls);
        r.passed = r.good_frac >= slo.good_frac_target;
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::RequestId;

    fn outcome(id: u64, ttft: f64, itl: f64) -> RequestOutcome {
        RequestOutcome {
            id: RequestId(id),
            arrival_ms: 0.0,
            first_token_ms: ttft,
            last_token_ms: ttft + itl * 199.0,
            prompt_tokens: 4000,
            output_tokens: 200,
            requested_tokens: 200,
        }
    }

    #[test]
    fn percentiles_are_nearest_rank() {
        let v: Vec<f64> = (1..=100).map(f64::from).collect();
        let s = LatencyStats::from_samples(&v);
        assert_eq!(s.p50, 50.0);
        assert_eq!(s.p99, 99.0);
        assert_eq!(s.max, 100.0);
    }

    #[test]
    fn a_request_is_scored_on_its_own_budgets_not_a_percentile() {
        let slo = Slo::default();
        // 9 clean requests and 1 with a 6 s TTFT: the p99 TTFT is awful, but
        // good_frac is 0.90 and the run passes. Judging this workload by a
        // percentile gate is the mistake this assertion exists to prevent.
        let mut outs: Vec<_> = (0..9).map(|i| outcome(i, 500.0, 17.0)).collect();
        outs.push(outcome(9, 6000.0, 17.0));
        let r = GoodputReport::from_outcomes(&outs, 10.0, &slo);
        assert_eq!(r.good_requests, 9);
        assert!((r.good_frac - 0.90).abs() < 1e-9);
        assert!(r.passed);
        assert_eq!(r.late_first_token, 1);
        assert!(r.ttft.p99 > slo.ttft_ms);
    }

    #[test]
    fn mean_itl_uses_gaps_not_tokens() {
        let o = outcome(0, 100.0, 20.0);
        assert!((o.mean_itl_ms() - 20.0).abs() < 1e-9);
    }
}
