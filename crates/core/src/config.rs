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
            // Derived from the model shape rather than restated, so the
            // capacity model and the simulator cannot disagree about how big
            // one request's KV is.
            kv_bytes_per_token: 2.0
                * f64::from(self.model.num_layers)
                * f64::from(self.model.num_kv_heads)
                * f64::from(self.model.head_dim),
            kv_xfer_gib_s: self.calibration.kv_xfer_gib_s,
            xfer_concurrency: self.kv.xfer_concurrency,
            weights_gib: self.model.weights_gib,
            gpu_gib: self.topology.gpu_gib,
            min_free_gib_per_rank: self.topology.min_free_gib_per_rank,
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
    /// Stored weights, GiB. 220.2 measured over the 24 FP8 safetensors shards
    /// of Qwen3-235B-A22B-Instruct-2507. This is what has to fit, and it is
    /// the stored count rather than the 22B active -- routing chooses which
    /// experts run, not which are resident.
    pub weights_gib: f64,
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
            weights_gib: 220.2,
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
    /// Usable GiB on one GPU after the driver's reservation. 131 on H200.
    pub gpu_gib: f64,
    /// GiB a rank needs beyond its weight shard. See
    /// `CapacityModel::min_free_gib_per_rank` -- bracketed by measurement at
    /// TP2 (21 GiB, unrunnable) and TP4 (76 GiB, runs), not observed directly.
    pub min_free_gib_per_rank: f64,
}

impl Default for TopologyConfig {
    /// 4P1D: four TP2 prefill workers feeding one TP8 decode worker. Narrow
    /// prefill workers cut the ring all-reduce traffic that profiling showed
    /// eating 25 % of prefill GPU time.
    fn default() -> Self {
        Self {
            total_gpus: 16,
            prefill_workers: 2,
            // 2P2D on TP4, the shape this deployment runs. It was 4P1D on
            // TP2 until TP2 was measured unrunnable -- 110.1 GiB of weights on
            // a 131 GiB card leaves 21 GiB, which cannot hold the activation
            // workspace, the CUDA graphs and a KV pool at once.
            prefill_tp: 4,
            decode_workers: 2,
            decode_tp: 4,
            gpu_gib: 131.0,
            min_free_gib_per_rank: 40.0,
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
    /// How many P/D KV transfers may be in flight at once, per decode worker.
    ///
    /// Sixteen, because that is what this deployment configures. UPSTREAM'S
    /// DEFAULT IS ONE, and the difference is the whole point: `mRecvBufferCount = getEnvRequestKVCacheConcurrent()
    /// ? getEnvKVCacheRecvBufferCount() : 1` (baseTransBuffer.cpp:109), and
    /// TRTLLM_REQUEST_KV_CACHE_CONCURRENT defaults to false, so the declared
    /// default of 2 on the count is never read. assignBufferIndex then blocks
    /// on a condition variable while the one buffer is taken.
    ///
    /// The simulator used to schedule the handoff as a fixed delay with no
    /// contention, which is why it could not reproduce job 316849's ceiling:
    /// serialised transfers of 181.85 ms cap the system at 5.50 req/s no matter
    /// how many GPUs are behind them, and 4.08 req/s was measured.
    ///
    /// Set this to 1 to model an unconfigured deployment; the simulator then
    /// reproduces the ceiling instead of predicting past it.
    pub xfer_concurrency: u32,
}

impl Default for KvConfig {
    fn default() -> Self {
        Self {
            block_size: 128,
            num_blocks: 8192,
            enable_prefix_cache: false,
            admission_watermark: 0.05,
            xfer_concurrency: 16,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct CalibrationConfig {
    pub prefill: PrefillCalibration,
    pub decode: DecodeCalibration,
    pub assumed_good_frac: f64,
    /// Effective P/D KV transfer bandwidth in GiB/s, per transfer.
    ///
    /// 2.1, from job 316849, and it is an UPPER bound on the fabric rather than
    /// a measurement of it. The model shape gives 0.359 GiB of KV for a
    /// 4000-token request (2 x 94 layers x 4 KV heads x 128 dim), and AIPerf
    /// measured Time to Second Token -- the gap between the prefill worker's
    /// first token and the decode worker's second -- at avg 181.85 ms with
    /// p50 191 and p99 208. 0.359 / 0.18185 = 1.97 GiB/s.
    ///
    /// That interval contains the transfer, whatever queue it waited in, and
    /// the first decode step, so the fabric may be faster than this and
    /// something else slower. What it is not is 40 GiB/s, which this simulator
    /// assumed and which made the handoff 19x too cheap to ever bind. A model
    /// that cannot represent the constraint cannot warn about it, and job
    /// 316849 spent 163 SU discovering one this simulator had already been
    /// asked about.
    ///
    /// Worth checking against the SGLang precedent recorded in
    /// scripts/stage-d-235b-disagg.sbatch: per-token RDMA there generated
    /// 4000 x 94 = ~376,000 small operations per request, which at 0.48 us each
    /// is the same 182 ms. If TensorRT-LLM does likewise, this constant is
    /// measuring operation count, not bandwidth, and will not improve with a
    /// faster fabric.
    pub kv_xfer_gib_s: f64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            prefill: PrefillCalibration::default(),
            decode: DecodeCalibration::default(),
            assumed_good_frac: 0.93,
            kv_xfer_gib_s: 2.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default describes the deployment that runs, and every part of it has
    /// been measured. It was 4P1D on TP2 prefill with a TP8 decode worker until
    /// both halves of that were refuted: TP2 leaves 21 GiB per rank and is
    /// unrunnable at any KV fraction, and one TP8 decode worker delivered
    /// 815 tok/s where two TP4 workers reach 2,170-2,470 -- Qwen3-235B has four
    /// KV heads, so TP4 gives each rank one and TP8 must duplicate.
    #[test]
    fn default_config_is_valid_and_is_2p2d_on_tp4() {
        let c = Config::default();
        c.validate().expect("default config must validate");
        assert_eq!(c.topology.prefill_workers, 2);
        assert_eq!(c.topology.prefill_tp, 4);
        assert_eq!(c.topology.decode_workers, 2);
        assert_eq!(c.topology.decode_tp, 4);
        // Every rank must have room for more than its weight shard.
        let m = c.capacity_model();
        assert!(m.fits_in_memory(c.topology.prefill_tp));
        assert!(m.fits_in_memory(c.topology.decode_tp));
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
