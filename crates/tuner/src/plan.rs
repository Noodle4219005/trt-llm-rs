//! Cross-checking and scoring AIConfigurator's candidates.

use serde::Serialize;
use trtllm_core::{config::Config, GoodputReport};
use trtllm_sim::{SimSetup, Simulator};

use crate::aic::AicCandidate;

pub trait CandidateEvaluator {
    fn evaluate(&self, config: &Config) -> GoodputReport;
}

/// Scores candidate configurations using the deterministic local simulator.
#[derive(Clone, Copy, Debug, Default)]
pub struct SimulationEvaluator;

impl CandidateEvaluator for SimulationEvaluator {
    fn evaluate(&self, config: &Config) -> GoodputReport {
        Simulator::new(SimSetup {
            config: config.clone(),
        })
        .run()
        .goodput
    }
}

/// Injects a score collected by one measured run without duplicating scoring.
pub struct MeasuredRunEvaluator<F> {
    measure: F,
}

impl<F> MeasuredRunEvaluator<F> {
    pub fn new(measure: F) -> Self {
        Self { measure }
    }
}

impl<F> CandidateEvaluator for MeasuredRunEvaluator<F>
where
    F: Fn(&Config) -> GoodputReport,
{
    fn evaluate(&self, config: &Config) -> GoodputReport {
        (self.measure)(config)
    }
}

/// AIConfigurator's prediction next to ours, for the same topology.
///
/// A disagreement is the point of this struct, not a failure of it. Their model
/// is a kernel-timing database for TensorRT-LLM; ours is two saturated
/// measurements of a different backend on this cluster. When they agree the
/// topology is probably safe. When they do not, the gap says which assumption
/// to go and measure - and that is worth more than either number alone.
#[derive(Clone, Debug, Serialize)]
pub struct CrossCheck {
    pub aic_output_tok_s_per_gpu: Option<f64>,
    pub our_output_tok_s_per_gpu: f64,
    pub ratio: Option<f64>,
    pub verdict: &'static str,
    pub note: String,
}

impl CrossCheck {
    pub fn build(candidate: &AicCandidate, cfg: &Config) -> Self {
        let m = cfg.capacity_model();
        let t = cfg.topology;
        let split = m.evaluate(
            t.prefill_workers * t.prefill_tp,
            t.decode_workers * t.decode_tp,
            t.prefill_tp,
            t.decode_tp,
        );
        let gpus = f64::from(t.total_gpus.max(1));
        let ours = split.sustainable_req_s * f64::from(cfg.workload.osl) / gpus;
        // Prefer AIConfigurator's own `seq/s`: it is the same quantity as our
        // sustainable request rate, so comparing them needs no conversion and
        // no assumption about which OSL either side divided by.
        let theirs = candidate
            .predicted_seq_s
            .map(|s| s * f64::from(cfg.workload.osl) / gpus)
            .or(candidate.predicted_tokens_s_per_gpu);
        let ratio = theirs.map(|x| x / ours.max(f64::MIN_POSITIVE));

        let (verdict, note) = match ratio {
            None => (
                "no-aic-number",
                "AIConfigurator did not report tokens/s/gpu".to_string(),
            ),
            Some(r) if (0.75..=1.333).contains(&r) => ("agree", format!("within 33 %: {r:.2}x")),
            Some(r) if r > 1.333 => (
                "aic-optimistic",
                format!(
                    "AIConfigurator predicts {r:.2}x our measured capacity; either its database \
                     covers a kernel path this cluster does not run, or our calibration is stale"
                ),
            ),
            Some(r) => (
                "aic-pessimistic",
                format!(
                    "AIConfigurator predicts {r:.2}x our measured capacity; check that the \
                     model/system/backend combination is inside its support matrix"
                ),
            ),
        };
        Self {
            aic_output_tok_s_per_gpu: theirs,
            our_output_tok_s_per_gpu: ours,
            ratio,
            verdict,
            note,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TuningRow {
    pub label: String,
    pub config: Config,
    pub cross_check: CrossCheck,
    pub score: GoodputReport,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TuningPlan {
    pub rows: Vec<TuningRow>,
    pub skipped: Vec<String>,
}

impl TuningPlan {
    /// Score every candidate under the metric we are actually judged on.
    ///
    /// `base` supplies everything AIConfigurator does not model: the SLO, the
    /// scheduler policy, the chunk size, the calibration. Only the topology
    /// comes from the candidate, which is the only part AIConfigurator is
    /// better placed to choose than we are.
    pub fn evaluate(candidates: &[AicCandidate], base: &Config) -> Self {
        Self::evaluate_with(candidates, base, &SimulationEvaluator)
    }

    pub fn evaluate_with<E: CandidateEvaluator>(
        candidates: &[AicCandidate],
        base: &Config,
        evaluator: &E,
    ) -> Self {
        let mut plan = Self::default();
        for c in candidates {
            let Some(cfg) = c.apply_to(base) else {
                plan.skipped.push(match c.mode {
                    crate::aic::DeploymentMode::Agg => format!(
                        "{}: aggregated layout, {} GPUs/worker - not a P/D split, listed for comparison only",
                        c.label(),
                        c.prefill_gpus_per_worker
                    ),
                    _ => format!("{}: unreadable topology in {}", c.label(), c.source),
                });
                continue;
            };
            if let Err(e) = cfg.validate() {
                plan.skipped.push(format!("{}: {e}", c.label()));
                continue;
            }
            let cross_check = CrossCheck::build(c, &cfg);
            let score = evaluator.evaluate(&cfg);
            plan.rows.push(TuningRow {
                label: c.label(),
                config: cfg,
                cross_check,
                score,
            });
        }
        plan.rows.sort_by(|a, b| {
            b.score
                .good_output_tok_s
                .total_cmp(&a.score.good_output_tok_s)
        });
        plan
    }

    pub fn best(&self) -> Option<&TuningRow> {
        self.rows.first()
    }

    pub fn table(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "{:<28} {:>10} {:>8} {:>9} {:>10} {:>16}\n",
            "topology", "goodput", "good%", "TTFT p99", "ITL mean", "vs aiconfigurator"
        ));
        for r in &self.rows {
            s.push_str(&format!(
                "{:<28} {:>10.2} {:>7.1}% {:>8.0}ms {:>9.2}ms {:>16}\n",
                r.label,
                r.score.good_output_tok_s,
                r.score.good_frac * 100.0,
                r.score.ttft.p99,
                r.score.itl.mean,
                r.cross_check.verdict,
            ));
        }
        for k in &self.skipped {
            s.push_str(&format!("skipped: {k}\n"));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aic::{candidates_from_table, DeploymentMode};
    use crate::csv::Table;

    fn short_base() -> Config {
        let mut c = Config::default();
        c.workload.warmup_s = 2.0;
        c.workload.benchmark_s = 6.0;
        c.workload.grace_s = 8.0;
        c.workload.concurrency = 48;
        c
    }

    /// Two rows in aiconfigurator's real disagg schema: 4x2 GPU prefill + 1x8
    /// GPU decode, and 2x4 GPU prefill + 1x8 GPU decode. The first row claims a
    /// far lower tokens/s/gpu than the second, so a ranking that follows the
    /// CSV's own numbers rather than our simulation would order them the other
    /// way round.
    fn two_rows() -> Vec<crate::aic::AicCandidate> {
        let csv = "(p)workers,(p)tp,(p)pp,(p)dp,(p)moe_ep,(d)workers,(d)tp,(d)pp,(d)dp,(d)moe_ep,num_total_gpus,seq/s,tokens/s/gpu\n\
4,1,1,2,2,1,8,1,1,8,16,14.375,179.685\n\
2,4,1,1,4,1,8,1,1,8,16,9.0,112.5\n";
        let t = Table::parse(csv).expect("parse");
        candidates_from_table(&t, DeploymentMode::Disagg, "test")
    }

    #[test]
    fn candidates_are_ranked_by_simulated_good_output_rate_not_by_the_csv_column() {
        let plan = TuningPlan::evaluate(&two_rows(), &short_base());
        assert_eq!(plan.rows.len(), 2, "{:?}", plan.skipped);
        for w in plan.rows.windows(2) {
            assert!(
                w[0].score.good_output_tok_s >= w[1].score.good_output_tok_s,
                "rows out of order:\n{}",
                plan.table()
            );
        }
        assert!(plan.table().contains("goodput"));
        // Deliberately NOT asserted: which shape wins. An earlier version
        // pinned the expected winner and the simulation disagreed for a real
        // reason - see `both_prefill_shapes_are_evaluated`.
    }

    #[test]
    fn measured_score_wins_even_when_its_request_rate_is_lower() {
        let evaluator = MeasuredRunEvaluator::new(|cfg: &Config| GoodputReport {
            req_per_s: if cfg.topology.prefill_tp == 2 {
                100.0
            } else {
                20.0
            },
            good_output_tok_s: if cfg.topology.prefill_tp == 2 {
                10.0
            } else {
                100.0
            },
            ..Default::default()
        });

        let plan = TuningPlan::evaluate_with(&two_rows(), &short_base(), &evaluator);

        assert_eq!(plan.rows.len(), 2, "{:?}", plan.skipped);
        assert_eq!(plan.rows[0].config.topology.prefill_tp, 4);
        assert_eq!(plan.rows[0].score.good_output_tok_s, 100.0);
        assert_eq!(plan.rows[0].score.req_per_s, 20.0);
        assert_eq!(plan.rows[1].score.req_per_s, 100.0);
    }

    /// Both shapes must be scored, on the topology the CSV actually describes.
    ///
    /// The winner depends on the *window*. On the full 60 s warmup / 120 s
    /// benchmark configuration the simulator ranks 4x2 GPU prefill above
    /// 2x4 GPU above 1x8 GPU, which is the all-reduce argument holding in
    /// steady state. On the short window used here, one wide worker can come
    /// out ahead instead, because a short window is dominated by the
    /// cold-start transient: every client arrives at t = 0 and one pooled
    /// queue drains a burst better than four separate ones however good the
    /// routing is. Both are real; they answer different questions.
    #[test]
    fn both_prefill_shapes_are_evaluated() {
        let plan = TuningPlan::evaluate(&two_rows(), &short_base());
        let shapes: Vec<u32> = plan
            .rows
            .iter()
            .map(|r| r.config.topology.prefill_tp)
            .collect();
        assert!(
            shapes.contains(&2) && shapes.contains(&4),
            "both shapes scored: {shapes:?}"
        );
        for r in &plan.rows {
            assert_eq!(r.config.topology.total_gpus, 16);
            assert!(
                r.score.total_requests > 0,
                "{} produced no scored requests",
                r.label
            );
        }
    }

    #[test]
    fn a_topology_that_does_not_fit_is_skipped_not_silently_resized() {
        let csv = "(p)workers,(p)tp,(p)pp,(p)dp,(d)workers,(d)tp,(d)pp,(d)dp,num_total_gpus\n\
4,8,1,1,1,8,1,1,16\n";
        let t = Table::parse(csv).expect("parse");
        let cands = candidates_from_table(&t, DeploymentMode::Disagg, "test");
        let plan = TuningPlan::evaluate(&cands, &short_base());
        assert!(
            plan.rows.is_empty(),
            "4x8 + 1x8 = 40 GPUs must not fit in 16"
        );
        assert_eq!(plan.skipped.len(), 1);
    }

    #[test]
    fn invalid_candidates_are_skipped_before_the_measured_evaluator_runs() {
        let csv = "(p)workers,(p)tp,(p)pp,(p)dp,(d)workers,(d)tp,(d)pp,(d)dp,num_total_gpus\n\n4,8,1,1,1,8,1,1,16\n";
        let t = Table::parse(csv).expect("parse");
        let candidates = candidates_from_table(&t, DeploymentMode::Disagg, "test");
        let evaluator = MeasuredRunEvaluator::new(|_: &Config| -> GoodputReport {
            panic!("invalid candidates must not be evaluated")
        });

        let plan = TuningPlan::evaluate_with(&candidates, &short_base(), &evaluator);

        assert!(plan.rows.is_empty());
        assert_eq!(plan.skipped.len(), 1);
    }

    #[test]
    fn a_wildly_optimistic_aic_number_is_flagged() {
        let csv =
            "(p)workers,(p)tp,(p)pp,(p)dp,(d)workers,(d)tp,(d)pp,(d)dp,num_total_gpus,seq/s\n\
4,1,1,2,1,8,1,1,16,9999.0\n";
        let t = Table::parse(csv).expect("parse");
        let cands = candidates_from_table(&t, DeploymentMode::Disagg, "test");
        let plan = TuningPlan::evaluate(&cands, &short_base());
        assert_eq!(plan.rows[0].cross_check.verdict, "aic-optimistic");
    }
}
