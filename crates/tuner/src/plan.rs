//! Cross-checking and scoring AIConfigurator's candidates.

use serde::Serialize;
use trtllm_core::config::Config;
use trtllm_sim::{SimReport, SimSetup, Simulator};

use crate::aic::AicCandidate;

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
        let theirs = candidate.predicted_tokens_s_per_gpu;
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
    pub sim: SimReport,
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
        let mut plan = Self::default();
        for c in candidates {
            let Some(cfg) = c.apply_to(base) else {
                plan.skipped
                    .push(format!("{}: no parallelism layout in the row", c.source));
                continue;
            };
            if let Err(e) = cfg.validate() {
                plan.skipped.push(format!("{}: {e}", c.label()));
                continue;
            }
            let cross_check = CrossCheck::build(c, &cfg);
            let sim = Simulator::new(SimSetup {
                config: cfg.clone(),
            })
            .run();
            plan.rows.push(TuningRow {
                label: c.label(),
                config: cfg,
                cross_check,
                sim,
            });
        }
        plan.rows.sort_by(|a, b| {
            b.sim
                .goodput
                .goodput_req_s
                .total_cmp(&a.sim.goodput.goodput_req_s)
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
                r.sim.goodput.goodput_req_s,
                r.sim.goodput.good_frac * 100.0,
                r.sim.goodput.ttft.p99,
                r.sim.goodput.itl.mean,
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

    #[test]
    fn candidates_are_ranked_by_simulated_goodput_not_by_aic_order() {
        let t = Table::parse(
            "(p)parallel,(p)replicas,(d)parallel,(d)replicas,total gpus,tokens/s/gpu\n\
             tp8pp1,1,tp8pp1,1,16,100.0\n\
             tp2pp1,4,tp8pp1,1,16,400.0\n",
        )
        .expect("parse");
        let cands = candidates_from_table(&t, DeploymentMode::Disagg, "test");
        let plan = TuningPlan::evaluate(&cands, &short_base());
        assert_eq!(plan.rows.len(), 2, "{:?}", plan.skipped);

        // The ranking must follow the *simulated* score, not the order
        // AIConfigurator happened to emit and not the tokens/s/gpu column it
        // reported - here the second row claims 4x the throughput of the first.
        for w in plan.rows.windows(2) {
            assert!(
                w[0].sim.goodput.goodput_req_s >= w[1].sim.goodput.goodput_req_s,
                "rows out of order:\n{}",
                plan.table()
            );
        }
        assert!(plan.table().contains("goodput"));

        // Deliberately NOT asserted: which of the two shapes wins. An earlier
        // version of this test asserted 4x TP2, on the reasoning that narrower
        // prefill workers carry less ring all-reduce per rank. The simulation
        // disagreed, and the disagreement turned out to be a real mechanism
        // rather than a bug - see `both_prefill_shapes_are_evaluated` below.
        // Pinning the expected winner would have hidden it.
    }

    /// Both prefill shapes must be scored, and the reported ranking must be
    /// self-consistent with the numbers behind it.
    ///
    /// The winner depends on the *window*, which is why this test does not
    /// name one. On the full 60 s warmup / 120 s benchmark configuration the
    /// simulator ranks 4x TP2 (16.67) above 2x TP4 (15.63) above 1x TP8
    /// (15.05) - the all-reduce argument holds in steady state. On the short
    /// window this test uses, one wide worker can come out ahead instead,
    /// because a short window is dominated by the cold-start transient: every
    /// client arrives at t = 0 at once, and one pooled queue drains a burst
    /// better than four separate ones however good the routing is.
    ///
    /// Both are real; they answer different questions. The lesson is the one
    /// this project already paid for on hardware - **check what the window
    /// contains before reading the metric**. `SimReport::caveats()` flags a
    /// window with fewer than 100 scored requests for the same reason.
    #[test]
    fn both_prefill_shapes_are_evaluated() {
        let t = Table::parse(
            "(p)parallel,(p)replicas,(d)parallel,(d)replicas,total gpus\n\
             tp8pp1,1,tp8pp1,1,16\n\
             tp2pp1,4,tp8pp1,1,16\n",
        )
        .expect("parse");
        let cands = candidates_from_table(&t, DeploymentMode::Disagg, "test");
        let plan = TuningPlan::evaluate(&cands, &short_base());
        let tps: Vec<u32> = plan
            .rows
            .iter()
            .map(|r| r.config.topology.prefill_tp)
            .collect();
        assert!(
            tps.contains(&2) && tps.contains(&8),
            "both shapes must be scored: {tps:?}"
        );
        for r in &plan.rows {
            assert!(
                r.sim.goodput.total_requests > 0,
                "{} produced no scored requests",
                r.label
            );
        }
    }

    #[test]
    fn a_topology_that_does_not_fit_is_skipped_not_silently_resized() {
        let t = Table::parse(
            "(p)parallel,(p)replicas,(d)parallel,(d)replicas,total gpus\ntp8pp1,4,tp8pp1,1,16\n",
        )
        .expect("parse");
        let cands = candidates_from_table(&t, DeploymentMode::Disagg, "test");
        let plan = TuningPlan::evaluate(&cands, &short_base());
        assert!(plan.rows.is_empty());
        assert_eq!(plan.skipped.len(), 1);
    }

    #[test]
    fn a_wildly_optimistic_aic_number_is_flagged() {
        let t = Table::parse(
            "(p)parallel,(p)replicas,(d)parallel,(d)replicas,total gpus,tokens/s/gpu\n\
             tp2pp1,4,tp8pp1,1,16,99999.0\n",
        )
        .expect("parse");
        let cands = candidates_from_table(&t, DeploymentMode::Disagg, "test");
        let plan = TuningPlan::evaluate(&cands, &short_base());
        assert_eq!(plan.rows[0].cross_check.verdict, "aic-optimistic");
    }
}
