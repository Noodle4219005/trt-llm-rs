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
    pub fn gpus(&self) -> u32 {
        self.tp.max(1) * self.pp.max(1)
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
    pub prefill_replicas: u32,
    pub decode_replicas: u32,
    pub total_gpus: u32,
    /// AIConfigurator's own predictions, for cross-checking only.
    pub predicted_tokens_s_per_gpu: Option<f64>,
    pub predicted_ttft_ms: Option<f64>,
    pub predicted_tpot_ms: Option<f64>,
    pub predicted_concurrency: Option<f64>,
    pub raw: BTreeMap<String, String>,
}

impl AicCandidate {
    /// Turn a candidate into a full deployment config by overlaying its
    /// topology onto `base`. Everything AIConfigurator does not model -
    /// scheduler policy, admission, chunk size - comes from `base` unchanged.
    pub fn apply_to(&self, base: &Config) -> Option<Config> {
        let p = self.prefill?;
        let d = self.decode?;
        let mut cfg = base.clone();
        cfg.topology.prefill_tp = p.tp.max(1);
        cfg.topology.decode_tp = d.tp.max(1);
        cfg.topology.prefill_workers = self.prefill_replicas.max(1);
        cfg.topology.decode_workers = self.decode_replicas.max(1);
        if self.total_gpus > 0 {
            cfg.topology.total_gpus = self.total_gpus;
        }
        Some(cfg)
    }

    pub fn label(&self) -> String {
        let p = self.prefill.map_or("?".to_string(), |s| {
            format!("{}xtp{}", self.prefill_replicas, s.tp)
        });
        let d = self.decode.map_or("?".to_string(), |s| {
            format!("{}xtp{}", self.decode_replicas, s.tp)
        });
        format!("{} P[{}] D[{}]", self.mode.as_str(), p, d)
    }
}

/// Extract candidates from an AIConfigurator CSV.
pub fn candidates_from_table(t: &Table, mode: DeploymentMode, source: &str) -> Vec<AicCandidate> {
    t.rows
        .iter()
        .map(|row| {
            let prefill = t
                .cell(row, &["(p)", "parallel"])
                .or_else(|| t.cell(row, &["prefill", "parallel"]))
                .or_else(|| {
                    if mode == DeploymentMode::Agg {
                        t.cell(row, &["parallel"])
                    } else {
                        None
                    }
                })
                .and_then(ParallelSpec::parse);
            let decode = t
                .cell(row, &["(d)", "parallel"])
                .or_else(|| t.cell(row, &["decode", "parallel"]))
                .or_else(|| {
                    if mode == DeploymentMode::Agg {
                        t.cell(row, &["parallel"])
                    } else {
                        None
                    }
                })
                .and_then(ParallelSpec::parse);

            let prefill_replicas = t
                .number(row, &["(p)", "replicas"])
                .or_else(|| t.number(row, &["prefill", "replicas"]))
                .unwrap_or(1.0) as u32;
            let decode_replicas = t
                .number(row, &["(d)", "replicas"])
                .or_else(|| t.number(row, &["decode", "replicas"]))
                .unwrap_or(1.0) as u32;

            let total_gpus = t
                .number(row, &["total", "gpus"])
                .map(|v| v as u32)
                .unwrap_or_else(|| {
                    prefill.map_or(0, |s| s.gpus() * prefill_replicas)
                        + decode.map_or(0, |s| s.gpus() * decode_replicas)
                });

            AicCandidate {
                source: source.to_string(),
                mode,
                prefill,
                decode,
                prefill_replicas,
                decode_replicas,
                total_gpus,
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
        assert_eq!(ParallelSpec::parse("tp2").map(|s| s.gpus()), Some(2));
        assert_eq!(ParallelSpec::parse("tp2pp2").map(|s| s.gpus()), Some(4));
        assert!(ParallelSpec::parse("").is_none());
        assert!(ParallelSpec::parse("nonsense").is_none());
    }

    #[test]
    fn a_disagg_row_becomes_a_topology() {
        let t = Table::parse(
            "(p)parallel,(p)replicas,(d)parallel,(d)replicas,total gpus,tokens/s/gpu,ttft,tpot\n\
             tp2pp1,4,tp8pp1,1,16,446.85,453.18,18.66\n",
        )
        .expect("parse");
        let c = &candidates_from_table(&t, DeploymentMode::Disagg, "test")[0];
        assert_eq!(c.prefill.expect("prefill").tp, 2);
        assert_eq!(c.prefill_replicas, 4);
        assert_eq!(c.decode.expect("decode").tp, 8);
        assert_eq!(c.decode_replicas, 1);
        assert_eq!(c.total_gpus, 16);

        let cfg = c.apply_to(&Config::default()).expect("config");
        assert_eq!(cfg.topology.prefill_workers, 4);
        assert_eq!(cfg.topology.prefill_tp, 2);
        assert_eq!(cfg.topology.decode_tp, 8);
        cfg.validate().expect("4P1D must fit in 16 GPUs");
    }

    #[test]
    fn total_gpus_is_derived_when_the_column_is_missing() {
        let t =
            Table::parse("(p)parallel,(p)replicas,(d)parallel,(d)replicas\ntp2pp1,4,tp8pp1,1\n")
                .expect("parse");
        let c = &candidates_from_table(&t, DeploymentMode::Disagg, "test")[0];
        assert_eq!(c.total_gpus, 16);
    }

    #[test]
    fn the_support_check_comes_before_the_search() {
        let run = AicRun::from_config(&Config::default(), "h200_sxm", "trtllm", "/tmp/aic");
        let support = run.shell(&run.support_command());
        assert!(support.contains("cli support"));
        let search = run.shell(&run.search_command());
        assert!(search.contains("--ttft 3000"));
        assert!(search.contains("--tpot 20.00"));
        assert!(search.contains("--total-gpus 16"));
    }
}
