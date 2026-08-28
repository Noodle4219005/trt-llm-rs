//! Talking to AIConfigurator.

use std::collections::BTreeMap;

use serde::Serialize;
use trtllm_core::config::Config;

use crate::csv::Table;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum DeploymentMode {
    Agg,
    Disagg,
}

impl DeploymentMode {
    pub fn as_str(self) -> &'static str {
        match self {
            DeploymentMode::Agg => "agg",
            DeploymentMode::Disagg => "disagg",
        }
    }
}

/// A parallelism layout as AIConfigurator spells it, e.g. `tp4pp1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ParallelSpec {
    pub tp: u32,
    pub pp: u32,
    pub ep: u32,
    pub dp: u32,
}

impl ParallelSpec {
    /// GPUs one worker of this layout occupies.
    ///
    /// `tp * pp * dp`. Data parallel counts: AIConfigurator's own arithmetic
    /// spells a 2-GPU prefill worker as `2 (=1x1x2)` for `tp1pp1dp2`. Leaving
    /// `dp` out makes every attention-DP layout look like a single GPU, which
    /// is how a 16-GPU recommendation gets read as a 2-GPU one.
    pub fn gpus(&self) -> u32 {
        self.tp.max(1) * self.pp.max(1) * self.dp.max(1)
    }

    /// Compact spelling, omitting degree-1 dimensions.
    pub fn spell(&self) -> String {
        let mut out = String::new();
        for (k, v) in [
            ("tp", self.tp),
            ("pp", self.pp),
            ("dp", self.dp),
            ("ep", self.ep),
        ] {
            if v > 1 {
                out.push_str(&format!("{k}{v}"));
            }
        }
        if out.is_empty() {
            out.push_str("tp1");
        }
        out
    }

    /// Parse `tp4pp1`, `tp8pp1ep8`, `TP2PP1DP1` and friends. Unknown keys are
    /// ignored rather than fatal - AIConfigurator may add dimensions.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim().to_ascii_lowercase();
        if s.is_empty() {
            return None;
        }
        let bytes = s.as_bytes();
        let mut out = ParallelSpec {
            tp: 1,
            pp: 1,
            ep: 1,
            dp: 1,
        };
        let mut i = 0;
        let mut saw_any = false;
        while i < bytes.len() {
            if i + 1 >= bytes.len() {
                break;
            }
            let key = &s[i..i + 2];
            let mut j = i + 2;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j == i + 2 {
                // Not a `xxN` group; give up rather than misparse.
                return if saw_any { Some(out) } else { None };
            }
            let value: u32 = s[i + 2..j].parse().ok()?;
            match key {
                "tp" => out.tp = value,
                "pp" => out.pp = value,
                "ep" => out.ep = value,
                "dp" => out.dp = value,
                _ => {}
            }
            saw_any = true;
            i = j;
        }
        if saw_any {
            Some(out)
        } else {
            None
        }
    }
}

/// One row of an AIConfigurator result table.
#[derive(Clone, Debug, Serialize)]
pub struct AicCandidate {
    pub source: String,
    pub mode: DeploymentMode,
    pub prefill: Option<ParallelSpec>,
    pub decode: Option<ParallelSpec>,
    pub prefill_workers: u32,
    pub decode_workers: u32,
    pub prefill_gpus_per_worker: u32,
    pub decode_gpus_per_worker: u32,
    pub total_gpus: u32,
    /// AIConfigurator's own predicted request rate, `seq/s` - directly
    /// comparable to our goodput denominator, unlike a tokens/s/gpu figure that
    /// has to be divided back out.
    pub predicted_seq_s: Option<f64>,
    /// AIConfigurator's own predictions, for cross-checking only.
    pub predicted_tokens_s_per_gpu: Option<f64>,
    pub predicted_ttft_ms: Option<f64>,
    pub predicted_tpot_ms: Option<f64>,
    pub predicted_concurrency: Option<f64>,
    pub raw: BTreeMap<String, String>,
}

impl AicCandidate {
    /// Overlay this candidate's topology onto `base`. Everything AIConfigurator
    /// does not model - scheduler policy, admission, chunk size - comes from
    /// `base` unchanged.
    ///
    /// Returns `None` for an aggregated candidate: this control plane is
    /// disaggregated by construction, and pretending an agg layout is a P/D
    /// split would score a deployment that does not exist. An agg row is
    /// information about AIConfigurator's opinion, not a config we can run.
    pub fn apply_to(&self, base: &Config) -> Option<Config> {
        if self.mode == DeploymentMode::Agg {
            return None;
        }
        if self.prefill_gpus_per_worker == 0
            || self.decode_gpus_per_worker == 0
            || self.prefill_workers == 0
            || self.decode_workers == 0
        {
            return None;
        }
        let mut cfg = base.clone();
        // `prefill_tp` in our config means GPUs per prefill worker. That is what
        // the capacity model needs, and it is *not* always a tensor parallel
        // degree - AIConfigurator's winner here is `tp1pp1dp2etp1ep2`: two GPUs
        // of attention data-parallel plus expert-parallel MoE, which carries no
        // attention all-reduce at all. Our `(t-1)/t` correction assumes TP and
        // therefore under-credits that layout.
        cfg.topology.prefill_tp = self.prefill_gpus_per_worker;
        cfg.topology.decode_tp = self.decode_gpus_per_worker;
        cfg.topology.prefill_workers = self.prefill_workers;
        cfg.topology.decode_workers = self.decode_workers;
        if self.total_gpus > 0 {
            cfg.topology.total_gpus = self.total_gpus;
        }
        Some(cfg)
    }

    pub fn label(&self) -> String {
        let spell = |n: u32, gpus: u32, s: Option<ParallelSpec>| match s {
            Some(s) => format!("{n}x{gpus}gpu:{}", s.spell()),
            None => format!("{n}x{gpus}gpu"),
        };
        format!(
            "{} P[{}] D[{}]",
            self.mode.as_str(),
            spell(
                self.prefill_workers,
                self.prefill_gpus_per_worker,
                self.prefill
            ),
            spell(
                self.decode_workers,
                self.decode_gpus_per_worker,
                self.decode
            )
        )
    }
}

/// Extract candidates from an AIConfigurator CSV.
///
/// Column names are the ones 0.7.0 actually writes, read off a real
/// `best_config_topn.csv` rather than from the docs:
///
/// ```text
/// model,isl,osl,prefix,concurrency,request_rate,
/// (p)bs,(p)global_bs,(p)workers,(d)bs,(d)global_bs,(d)workers,
/// ttft,tpot,request_latency,seq/s,seq/s/gpu,tokens/s,tokens/s/gpu,tokens/s/user,
/// (p)seq/s/worker,(d)seq/s/worker,num_total_gpus,
/// (p)tp,(p)pp,(p)dp,(p)moe_tp,(p)moe_ep,(p)parallel,...
/// ```
///
/// Two things this gets right that the first version did not. The numeric
/// `(p)tp/(p)pp/(p)dp` columns are preferred over parsing the `(p)parallel`
/// string. And a row whose worker count cannot be read is **rejected**, not
/// defaulted to 1 - defaulting is what turned a 4-worker, 16-GPU
/// recommendation into a 1-worker, 2-GPU one, producing a goodput 6x too low
/// that still looked like a number.
pub fn candidates_from_table(t: &Table, mode: DeploymentMode, source: &str) -> Vec<AicCandidate> {
    t.rows
        .iter()
        .map(|row| {
            let spec = |side: &str| -> Option<ParallelSpec> {
                let n = |k: &str| t.number(row, &[side, k]).map(|v| (v as u32).max(1));
                match (n("tp"), n("pp")) {
                    (Some(tp), Some(pp)) => Some(ParallelSpec {
                        tp,
                        pp,
                        dp: n("dp").unwrap_or(1),
                        ep: n("moe_ep").unwrap_or(1),
                    }),
                    _ => t
                        .cell(row, &[side, "parallel"])
                        .and_then(ParallelSpec::parse),
                }
            };

            let (prefill, decode, pw, dw) = match mode {
                DeploymentMode::Disagg => (
                    spec("(p)"),
                    spec("(d)"),
                    t.number(row, &["(p)", "workers"]),
                    t.number(row, &["(d)", "workers"]),
                ),
                // An aggregated row has one worker kind doing both phases, and
                // its schema carries **no worker or replica column at all**:
                // `model,isl,osl,prefix,concurrency,request_rate,bs,global_bs,
                //  ttft,tpot,request_latency,seq/s,...,num_total_gpus,
                //  tp,pp,dp,moe_tp,moe_ep,parallel,...`
                // So the replica count is `num_total_gpus / gpus_per_worker`.
                DeploymentMode::Agg => {
                    let one = t.cell(row, &["parallel"]).and_then(ParallelSpec::parse);
                    let gpus = one.map_or(0, |x: ParallelSpec| x.gpus());
                    let n = t
                        .number(row, &["replicas"])
                        .or_else(|| t.number(row, &["workers"]))
                        .or_else(|| {
                            t.number(row, &["num_total_gpus"])
                                .filter(|_| gpus > 0)
                                .map(|total| (total / f64::from(gpus)).floor())
                        });
                    (one, one, n, n)
                }
            };

            AicCandidate {
                source: source.to_string(),
                mode,
                prefill,
                decode,
                prefill_workers: pw.unwrap_or(0.0) as u32,
                decode_workers: dw.unwrap_or(0.0) as u32,
                prefill_gpus_per_worker: prefill.map_or(0, |s| s.gpus()),
                decode_gpus_per_worker: decode.map_or(0, |s| s.gpus()),
                total_gpus: t
                    .number(row, &["num_total_gpus"])
                    .or_else(|| t.number(row, &["total", "gpus"]))
                    .unwrap_or(0.0) as u32,
                predicted_seq_s: t.number(row, &["seq/s"]),
                predicted_tokens_s_per_gpu: t.number(row, &["tokens/s/gpu"]),
                predicted_ttft_ms: t.number(row, &["ttft"]),
                predicted_tpot_ms: t.number(row, &["tpot"]),
                predicted_concurrency: t.number(row, &["concurrency"]),
                raw: t.row_map(row),
            }
        })
        .collect()
}

/// The AIConfigurator invocation for a deployment, built but not run.
///
/// Printing the command instead of shelling out is deliberate: this is an
/// external tool with its own install requirements and its own support matrix,
/// and a silent fallback to weaker modelling is exactly the failure this
/// project must not paper over. Run it, read the support check, then point
/// [`crate::load_candidates`] at the `--save-dir`.
#[derive(Clone, Debug)]
pub struct AicRun {
    pub model_path: String,
    pub system: String,
    pub backend: String,
    pub backend_version: Option<String>,
    pub total_gpus: u32,
    pub isl: u32,
    pub osl: u32,
    pub ttft_ms: f64,
    /// AIConfigurator's SLA knob is time *per output token*. Our budget is mean
    /// inter-token latency, which is the same quantity under a different name -
    /// but ours is a per-request average with a 90 % pass threshold behind it,
    /// so passing our number here filters on the mean and nothing more.
    pub tpot_ms: f64,
    pub save_dir: String,
    pub database_mode: Option<String>,
    pub deployment_target: Option<String>,
}

impl AicRun {
    pub fn from_config(cfg: &Config, system: &str, backend: &str, save_dir: &str) -> Self {
        Self {
            model_path: cfg.model.name.clone(),
            system: system.to_string(),
            backend: backend.to_string(),
            backend_version: None,
            total_gpus: cfg.topology.total_gpus,
            isl: cfg.workload.isl,
            osl: cfg.workload.osl,
            ttft_ms: cfg.slo.ttft_ms,
            tpot_ms: cfg.slo.itl_ms,
            save_dir: save_dir.to_string(),
            database_mode: Some("SILICON".into()),
            deployment_target: None,
        }
    }

    /// `aiconfigurator cli support ...` - run this first. Outside the support
    /// matrix the numbers below are not estimates of anything.
    pub fn support_command(&self) -> Vec<String> {
        vec![
            "aiconfigurator".into(),
            "cli".into(),
            "support".into(),
            "--model-path".into(),
            self.model_path.clone(),
            "--system".into(),
            self.system.clone(),
            "--backend".into(),
            self.backend.clone(),
        ]
    }

    pub fn search_command(&self) -> Vec<String> {
        let mut v = vec![
            "aiconfigurator".into(),
            "cli".into(),
            "default".into(),
            "--model-path".into(),
            self.model_path.clone(),
            "--system".into(),
            self.system.clone(),
            "--backend".into(),
            self.backend.clone(),
            "--total-gpus".into(),
            self.total_gpus.to_string(),
            "--isl".into(),
            self.isl.to_string(),
            "--osl".into(),
            self.osl.to_string(),
            "--ttft".into(),
            format!("{:.0}", self.ttft_ms),
            "--tpot".into(),
            format!("{:.2}", self.tpot_ms),
            "--save-dir".into(),
            self.save_dir.clone(),
        ];
        if let Some(bv) = &self.backend_version {
            v.push("--backend-version".into());
            v.push(bv.clone());
        }
        if let Some(db) = &self.database_mode {
            v.push("--database-mode".into());
            v.push(db.clone());
        }
        if let Some(t) = &self.deployment_target {
            v.push("--deployment-target".into());
            v.push(t.clone());
        }
        v
    }

    pub fn shell(&self, argv: &[String]) -> String {
        argv.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real header and row, copied from
    /// `disagg/best_config_topn.csv` produced by aiconfigurator 0.7.0 for
    /// Qwen3-235B-A22B-FP8 on h200_sxm, trtllm, 16 GPUs, ISL 4000 / OSL 200.
    const DISAGG_CSV: &str = "model,isl,osl,prefix,concurrency,request_rate,\
(p)bs,(p)global_bs,(p)workers,(d)bs,(d)global_bs,(d)workers,\
ttft,tpot,request_latency,seq/s,seq/s/gpu,tokens/s,tokens/s/gpu,tokens/s/user,\
(p)seq/s/worker,(d)seq/s/worker,num_total_gpus,\
(p)tp,(p)pp,(p)dp,(p)moe_tp,(p)moe_ep,(p)parallel,(d)tp,(d)pp,(d)dp,(d)moe_tp,(d)moe_ep,(d)parallel\n\
Qwen/Qwen3-235B-A22B-FP8,4000,200,0,64,14.375,\
1,2,4,64,64,1,\
901.561,19.779,4837.582,14.375,0.898,2874.96,179.685,50.559,\
3.993,16.26,16,\
1,1,2,1,2,tp1pp1dp2etp1ep2,8,1,1,1,8,tp8pp1dp1etp1ep8\n";

    #[test]
    fn parallel_specs_round_trip() {
        assert_eq!(
            ParallelSpec::parse("tp4pp1"),
            Some(ParallelSpec {
                tp: 4,
                pp: 1,
                ep: 1,
                dp: 1
            })
        );
        assert_eq!(
            ParallelSpec::parse("TP8PP1EP8"),
            Some(ParallelSpec {
                tp: 8,
                pp: 1,
                ep: 8,
                dp: 1
            })
        );
        assert!(ParallelSpec::parse("").is_none());
        assert!(ParallelSpec::parse("nonsense").is_none());
    }

    /// Data parallel multiplies the GPU count. Omitting it is how a two-GPU
    /// worker gets counted as one.
    #[test]
    fn gpus_per_worker_counts_dp() {
        assert_eq!(ParallelSpec::parse("tp1pp1dp2").map(|s| s.gpus()), Some(2));
        assert_eq!(ParallelSpec::parse("tp8pp1dp1").map(|s| s.gpus()), Some(8));
        assert_eq!(ParallelSpec::parse("tp2pp2").map(|s| s.gpus()), Some(4));
    }

    /// The real recommendation must come back as 4 prefill workers of 2 GPUs
    /// plus 1 decode worker of 8 GPUs, totalling 16 - not as one worker each.
    #[test]
    fn the_real_disagg_row_parses_as_4p1d_on_16_gpus() {
        let t = Table::parse(DISAGG_CSV).expect("parse");
        let c = &candidates_from_table(&t, DeploymentMode::Disagg, "test")[0];

        assert_eq!(c.prefill_workers, 4);
        assert_eq!(c.prefill_gpus_per_worker, 2, "tp1 x pp1 x dp2");
        assert_eq!(c.decode_workers, 1);
        assert_eq!(c.decode_gpus_per_worker, 8, "tp8 x pp1 x dp1");
        assert_eq!(c.total_gpus, 16);
        assert_eq!(
            c.prefill_workers * c.prefill_gpus_per_worker
                + c.decode_workers * c.decode_gpus_per_worker,
            c.total_gpus,
            "the GPU arithmetic has to close"
        );
        assert_eq!(c.predicted_seq_s, Some(14.375));
        assert_eq!(c.predicted_tpot_ms, Some(19.779));

        let cfg = c.apply_to(&Config::default()).expect("config");
        assert_eq!(cfg.topology.prefill_workers, 4);
        assert_eq!(cfg.topology.prefill_tp, 2);
        assert_eq!(cfg.topology.decode_tp, 8);
        cfg.validate().expect("4P1D must fit in 16 GPUs");
    }

    /// A row whose worker count is unreadable must be rejected, not silently
    /// treated as one worker.
    #[test]
    fn a_row_without_worker_counts_is_rejected() {
        let t = Table::parse("(p)parallel,(d)parallel\ntp1pp1dp2,tp8pp1dp1\n").expect("parse");
        let c = &candidates_from_table(&t, DeploymentMode::Disagg, "test")[0];
        assert_eq!(c.prefill_workers, 0);
        assert!(c.apply_to(&Config::default()).is_none());
    }

    /// The agg schema has no worker column; the replica count has to come from
    /// `num_total_gpus / gpus_per_worker`.
    #[test]
    fn agg_replicas_are_derived_from_the_gpu_total() {
        let csv =
            "num_total_gpus,tp,pp,dp,moe_ep,parallel,seq/s\n16,1,1,4,4,tp1pp1dp4etp1ep4,7.06\n";
        let t = Table::parse(csv).expect("parse");
        let c = &candidates_from_table(&t, DeploymentMode::Agg, "test")[0];
        assert_eq!(c.prefill_gpus_per_worker, 4, "tp1 x pp1 x dp4");
        assert_eq!(c.prefill_workers, 4, "16 GPUs / 4 per worker");
        // Still not something this disaggregated control plane can run.
        assert!(c.apply_to(&Config::default()).is_none());
        assert!(c.label().starts_with("agg "));
    }

    #[test]
    fn the_support_check_comes_before_the_search() {
        let run = AicRun::from_config(&Config::default(), "h200_sxm", "trtllm", "/tmp/aic");
        assert!(run.shell(&run.support_command()).contains("cli support"));
        let search = run.shell(&run.search_command());
        assert!(search.contains("--ttft 3000"));
        assert!(search.contains("--tpot 20.00"));
        assert!(search.contains("--total-gpus 16"));
    }
}
