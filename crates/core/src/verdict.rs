//! Compare what a run was predicted to do with what it did.
//!
//! Six 16-GPU jobs were interpreted against a model that had never once been
//! printed beside a result. The launcher now prints its prediction before the
//! workers start; this reads the AIPerf export afterwards and says whether the
//! two agree, and when they do not, which resource the gap implicates.
//!
//! The diagnosis matters more than the number. Job 316849 measured goodput 0.00
//! against a predicted 14.30, and the useful part was never "off by 14.30" --
//! it was "TTFT passed and ITL missed by 4.6x, so look at decode", which took
//! a person reading a console table to notice.

use serde::{Deserialize, Serialize};

use crate::capacity::Bottleneck;
use crate::slo::Slo;

/// The subset of AIPerf's export this needs. Everything else is ignored, so a
/// schema addition upstream does not break parsing.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct MeasuredRun {
    pub goodput_req_s: f64,
    pub request_throughput_req_s: f64,
    pub output_token_throughput: f64,
    pub ttft_avg_ms: f64,
    pub ttft_p90_ms: f64,
    pub itl_avg_ms: f64,
    pub request_count: f64,
}

/// Why a run missed, in the terms the gates are written in.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Diagnosis {
    /// Within tolerance of the prediction.
    Met,
    /// Requests were served but too slowly between tokens.
    ItlGate { measured_ms: f64, budget_ms: f64 },
    /// Requests waited too long for their first token.
    TtftGate { measured_ms: f64, budget_ms: f64 },
    /// Both gates passed and the rate still fell short, which points at a
    /// resource rather than a latency.
    ThroughputShortfall {
        measured_req_s: f64,
        predicted_req_s: f64,
        implicates: Bottleneck,
    },
    /// Nothing completed. Distinct from a shortfall: a run that served zero
    /// requests has not measured the deployment, it has measured a failure.
    NothingServed,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Verdict {
    pub predicted_goodput_req_s: f64,
    pub measured: MeasuredRun,
    /// measured / predicted. 1.0 is agreement.
    pub ratio: f64,
    pub diagnosis: Diagnosis,
}

impl Verdict {
    /// `tolerance` is the fraction by which measured may fall short and still
    /// count as agreement. 0.15 is generous on purpose: the point is to catch
    /// a model that is wrong about the mechanism, not one that is 8% optimistic.
    pub fn assess(
        measured: MeasuredRun,
        predicted_goodput_req_s: f64,
        slo: &Slo,
        binding: Bottleneck,
        tolerance: f64,
    ) -> Self {
        let ratio = if predicted_goodput_req_s > 0.0 {
            measured.goodput_req_s / predicted_goodput_req_s
        } else {
            f64::NAN
        };

        let diagnosis = if measured.request_count < 1.0 {
            Diagnosis::NothingServed
        } else if ratio >= 1.0 - tolerance {
            Diagnosis::Met
        } else if measured.itl_avg_ms > slo.itl_ms {
            // Checked before TTFT because ITL is charged per token and so
            // dominates a request's fate once it is missed at all.
            Diagnosis::ItlGate {
                measured_ms: measured.itl_avg_ms,
                budget_ms: slo.itl_ms,
            }
        } else if measured.ttft_p90_ms > slo.ttft_ms {
            Diagnosis::TtftGate {
                measured_ms: measured.ttft_p90_ms,
                budget_ms: slo.ttft_ms,
            }
        } else {
            Diagnosis::ThroughputShortfall {
                measured_req_s: measured.request_throughput_req_s,
                predicted_req_s: predicted_goodput_req_s,
                implicates: binding,
            }
        };

        Verdict {
            predicted_goodput_req_s,
            measured,
            ratio,
            diagnosis,
        }
    }

    /// One line, for a log a person will skim.
    pub fn summary(&self) -> String {
        match self.diagnosis {
            Diagnosis::Met => format!(
                "MET: {:.2} against {:.2} req/s predicted ({:.0}%)",
                self.measured.goodput_req_s,
                self.predicted_goodput_req_s,
                self.ratio * 100.0
            ),
            Diagnosis::NothingServed => {
                "NOTHING SERVED: this measured a failure, not the deployment".into()
            }
            Diagnosis::ItlGate {
                measured_ms,
                budget_ms,
            } => format!(
                "ITL GATE: {measured_ms:.1} ms against a {budget_ms:.0} ms budget \
                 ({:.1}x over). Requests were served and were too slow between \
                 tokens; TTFT was not the problem.",
                measured_ms / budget_ms
            ),
            Diagnosis::TtftGate {
                measured_ms,
                budget_ms,
            } => format!(
                "TTFT GATE: p90 {measured_ms:.0} ms against a {budget_ms:.0} ms \
                 budget. Tokens came fast enough once they started."
            ),
            Diagnosis::ThroughputShortfall {
                measured_req_s,
                predicted_req_s,
                implicates,
            } => format!(
                "SHORTFALL: {measured_req_s:.2} against {predicted_req_s:.2} req/s \
                 with both gates passing, which points at {implicates:?} rather \
                 than at a latency."
            ),
        }
    }
}

impl MeasuredRun {
    /// Parse AIPerf's `profile_export_aiperf.json`.
    ///
    /// Every metric there is `{"unit": ..., "avg": ..., "p90": ...}`, and a
    /// missing one is an error rather than a zero: a verdict built from an
    /// absent number is worse than no verdict.
    pub fn from_aiperf_json(doc: &serde_json::Value) -> Result<Self, String> {
        let field = |name: &str, stat: &str| -> Result<f64, String> {
            doc.get(name)
                .and_then(|m| m.get(stat))
                .and_then(|v| v.as_f64())
                .ok_or_else(|| format!("{name}.{stat} is missing from the export"))
        };
        Ok(MeasuredRun {
            goodput_req_s: field("goodput", "avg")?,
            request_throughput_req_s: field("request_throughput", "avg")?,
            output_token_throughput: field("output_token_throughput", "avg")?,
            ttft_avg_ms: field("time_to_first_token", "avg")?,
            ttft_p90_ms: field("time_to_first_token", "p90")?,
            itl_avg_ms: field("inter_token_latency", "avg")?,
            request_count: field("request_count", "avg")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Job 316849's export, verbatim, as a regression on the diagnosis.
    fn job_316849() -> MeasuredRun {
        MeasuredRun {
            goodput_req_s: 0.0,
            request_throughput_req_s: 4.077_099_667_664_089,
            output_token_throughput: 815.412_062_684_038_6,
            ttft_avg_ms: 995.785_916_810_810_7,
            ttft_p90_ms: 2980.90,
            itl_avg_ms: 91.144_873_879_169_5,
            request_count: 518.0,
        }
    }

    /// The run that started all of this. The number is not the useful part --
    /// "off by 14.30" says nothing. "TTFT passed, ITL missed by 4.6x, look at
    /// decode" is what a person had to read a console table to work out.
    #[test]
    fn the_first_complete_run_is_diagnosed_as_the_itl_gate() {
        let v = Verdict::assess(
            job_316849(),
            14.30,
            &Slo::default(),
            Bottleneck::Decode,
            0.15,
        );
        match v.diagnosis {
            Diagnosis::ItlGate {
                measured_ms,
                budget_ms,
            } => {
                assert!((measured_ms - 91.14).abs() < 0.01);
                assert!((budget_ms - 20.0).abs() < 0.01);
            }
            other => panic!("expected the ITL gate, got {other:?}"),
        }
        assert!(
            v.summary().contains("TTFT was not the problem"),
            "the summary must say which gate held: {}",
            v.summary()
        );
    }

    /// TTFT p90 was 2980.90 against 3000 -- inside by 19 ms. A diagnosis that
    /// blamed TTFT here would be pointing at the one thing that passed.
    #[test]
    fn the_gate_that_passed_is_not_blamed() {
        let v = Verdict::assess(
            job_316849(),
            14.30,
            &Slo::default(),
            Bottleneck::Decode,
            0.15,
        );
        assert!(!matches!(v.diagnosis, Diagnosis::TtftGate { .. }));
    }

    #[test]
    fn a_run_that_meets_its_prediction_says_so() {
        let mut m = job_316849();
        m.goodput_req_s = 13.5;
        m.itl_avg_ms = 16.0;
        let v = Verdict::assess(m, 14.30, &Slo::default(), Bottleneck::Decode, 0.15);
        assert_eq!(v.diagnosis, Diagnosis::Met);
    }

    /// Both gates passing and the rate still short is a different failure and
    /// has to read differently: it points at a resource, not a latency.
    #[test]
    fn passing_both_gates_and_still_falling_short_implicates_a_resource() {
        let mut m = job_316849();
        m.goodput_req_s = 5.0;
        m.request_throughput_req_s = 5.0;
        m.itl_avg_ms = 16.0;
        m.ttft_p90_ms = 1500.0;
        let v = Verdict::assess(m, 14.30, &Slo::default(), Bottleneck::KvTransfer, 0.15);
        match v.diagnosis {
            Diagnosis::ThroughputShortfall { implicates, .. } => {
                assert_eq!(implicates, Bottleneck::KvTransfer);
            }
            other => panic!("expected a shortfall, got {other:?}"),
        }
    }

    /// Zero completions is not a slow deployment, it is a broken one, and the
    /// four invalid disagg runs were all of this kind.
    #[test]
    fn zero_requests_is_a_failure_not_a_shortfall() {
        let mut m = job_316849();
        m.request_count = 0.0;
        let v = Verdict::assess(m, 14.30, &Slo::default(), Bottleneck::Decode, 0.15);
        assert_eq!(v.diagnosis, Diagnosis::NothingServed);
    }

    #[test]
    fn a_missing_metric_is_an_error_not_a_zero() {
        let doc: serde_json::Value = serde_json::from_str(r#"{"goodput": {"avg": 1.0}}"#).unwrap();
        let err = MeasuredRun::from_aiperf_json(&doc).expect_err("should not parse");
        assert!(err.contains("request_throughput"), "{err}");
    }
}
