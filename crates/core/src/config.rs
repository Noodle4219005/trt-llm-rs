//! Deployment configuration.
//!
//! One TOML file describes the whole deployment: the workload it is being
//! tuned for, the topology, the scheduler policy and the calibration constants
//! the capacity model reasons with. The simulator and the real workers read
//! the *same* file, so a policy proven in simulation is the policy that ships.

use serde::{Deserialize, Serialize};

use crate::capacity::{CapacityModel, DecodeCalibration, PrefillCalibration};
use crate::error::{Error, Result};
use crate::slo::Slo;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub model: ModelConfig,
    pub workload: WorkloadConfig,
    pub topology: TopologyConfig,
    pub scheduler: SchedulerConfig,
    pub kv: KvConfig,
    pub calibration: CalibrationConfig,
    pub slo: Slo,
}

impl Config {
    pub fn from_toml_str(s: &str) -> Result<Self> {
        toml::from_str(s).map_err(|e| Error::Config(e.to_string()))
    }

    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
        Self::from_toml_str(&text)
    }

    pub fn capacity_model(&self) -> CapacityModel {
        CapacityModel {
            isl: self.workload.isl,
            osl: self.workload.osl,
            slo: self.slo,
            prefill: self.calibration.prefill,
            decode: self.calibration.decode,
            good_frac: self.calibration.assumed_good_frac,
            itl_safety: self.scheduler.itl_safety,
        }
    }

    pub fn validate(&self) -> Result<()> {
        let t = &self.topology;
        if t.prefill_workers == 0 || t.decode_workers == 0 {
            return Err(Error::Config(
                "need at least one prefill and one decode worker".into(),
            ));
        }
        let used = t.prefill_workers * t.prefill_tp + t.decode_workers * t.decode_tp;
        if used > t.total_gpus {
            return Err(Error::Config(format!(
                "topology needs {used} GPUs but only {} are available",
                t.total_gpus
            )));
        }
        if self.slo.itl_ms <= 0.0 || self.slo.ttft_ms <= 0.0 {
            return Err(Error::Config("SLO budgets must be positive".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    pub name: String,
    pub tokenizer: String,
    /// FP8 is mandatory for the scored run; recorded here so a mis-set weight
    /// dtype shows up in the config dump rather than in the results.
    pub dtype: String,
    pub kv_dtype: String,
    pub num_layers: u32,
    pub hidden_size: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: "Qwen/Qwen3-235B-A22B-Instruct-2507".into(),
            tokenizer: "Qwen/Qwen3-235B-A22B-Instruct-2507".into(),
            dtype: "fp8".into(),
            kv_dtype: "fp8".into(),
            num_layers: 94,
            hidden_size: 4096,
            num_kv_heads: 4,
            head_dim: 128,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkloadConfig {
    /// Input sequence length. The scored workload uses stddev 0.
    pub isl: u32,
    /// Output sequence length, also stddev 0.
    pub osl: u32,
    /// Closed-loop client concurrency.
    pub concurrency: u32,
    pub warmup_s: f64,
    pub benchmark_s: f64,
    /// Extra time after the benchmark window in which requests issued inside
    /// the window may still finish and be counted.
    pub grace_s: f64,
    pub seed: u64,
    /// The scored run busts the prefix cache, so prefix reuse must not be part
    /// of any result we believe.
    pub cache_bust: bool,
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        Self {
            isl: 4000,
            osl: 200,
            concurrency: 80,
            warmup_s: 60.0,
            benchmark_s: 120.0,
            grace_s: 30.0,
            seed: 2026,
            cache_bust: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TopologyConfig {
    pub total_gpus: u32,
    pub prefill_workers: u32,
    pub prefill_tp: u32,
    pub decode_workers: u32,
    pub decode_tp: u32,
}

impl Default for TopologyConfig {
    /// 4P1D: four TP2 prefill workers feeding one TP8 decode worker. Narrow
    /// prefill workers cut the ring all-reduce traffic that profiling showed
    /// eating 25 % of prefill GPU time.
    fn default() -> Self {
        Self {
            total_gpus: 16,
            prefill_workers: 4,
            prefill_tp: 2,
            decode_workers: 1,
            decode_tp: 8,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SchedulerConfig {
    /// Tokens per prefill forward pass. 16384 lets 4-5 sequences share a batch
    /// on this model, worth ~11.6 % over the 4096 setting that forces one
    /// sequence per batch.
    pub chunked_prefill_tokens: u32,
    /// Upper bound on sequences co-scheduled in one prefill batch.
    pub max_prefill_seqs: u32,
    /// Hard cap on decode batch size, independent of the SLO-derived cap.
    pub max_decode_seqs: u32,
    /// Headroom multiplier applied to the ITL budget when sizing decode
    /// concurrency. 1.0 spends the whole budget; 0.9 keeps 10 % back.
    pub itl_safety: f64,
    /// Prefill ordering policy.
    pub prefill_policy: PrefillPolicy,
    /// Requests whose first-token deadline is already unreachable get moved to
    /// a background lane instead of holding the head of the queue.
    pub demote_hopeless: bool,
    /// Scheduler tick, milliseconds. The decode loop must run well inside the
    /// ITL budget; 1 ms is comfortable in Rust and was not in Python.
    pub tick_ms: f64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            chunked_prefill_tokens: 16384,
            max_prefill_seqs: 8,
            max_decode_seqs: 4096,
            itl_safety: 1.0,
            prefill_policy: PrefillPolicy::MooreHodgson,
            demote_hopeless: true,
            tick_ms: 1.0,
        }
    }
}

/// How the prefill queue is ordered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PrefillPolicy {
    /// First come, first served - what every stock serving stack does.
    Fcfs,
    /// Earliest deadline first. Optimal for minimising the *worst* lateness,
    /// which is not the metric here.
    Edf,
    /// Moore-Hodgson: maximise the number of requests that meet their
    /// first-token deadline. This is the metric here, and the algorithm is
    /// optimal for it on a single machine.
    MooreHodgson,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct KvConfig {
    /// Tokens per KV block. 128 measured best on this model.
    pub block_size: u32,
    /// Total blocks in the pool, per worker.
    pub num_blocks: u32,
    /// Enable the radix prefix cache. Off by default: the scored run busts the
    /// prefix, so leaving it on only buys a number that will not reproduce.
    pub enable_prefix_cache: bool,
    /// Fraction of the pool that must stay free before a new sequence is
    /// admitted, so an in-flight sequence never gets preempted mid-decode.
    pub admission_watermark: f64,
}

impl Default for KvConfig {
    fn default() -> Self {
        Self {
            block_size: 128,
            num_blocks: 8192,
            enable_prefix_cache: false,
            admission_watermark: 0.05,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct CalibrationConfig {
    pub prefill: PrefillCalibration,
    pub decode: DecodeCalibration,
    pub assumed_good_frac: f64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            prefill: PrefillCalibration::default(),
            decode: DecodeCalibration::default(),
            assumed_good_frac: 0.93,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid_and_is_4p1d() {
        let c = Config::default();
        c.validate().expect("default config must validate");
        assert_eq!(c.topology.prefill_workers, 4);
        assert_eq!(c.topology.prefill_tp, 2);
        assert_eq!(c.topology.decode_tp, 8);
        assert_eq!(
            c.topology.prefill_workers * c.topology.prefill_tp
                + c.topology.decode_workers * c.topology.decode_tp,
            16
        );
    }

    #[test]
    fn overcommitting_gpus_is_rejected() {
        let mut c = Config::default();
        c.topology.prefill_workers = 8;
        assert!(c.validate().is_err());
    }

    #[test]
    fn partial_toml_keeps_defaults() {
        let c = Config::from_toml_str("[workload]\nconcurrency = 96\n").expect("parse");
        assert_eq!(c.workload.concurrency, 96);
        assert_eq!(c.workload.isl, 4000);
        assert_eq!(c.slo.ttft_ms, 3000.0);
    }
}
