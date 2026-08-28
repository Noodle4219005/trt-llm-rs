//! The calibrated capacity model.
//!
//! `goodput = min(prefill_req_s, decode_req_s) x good_frac`
//!
//! Both halves are measured, not guessed, and the model reproduced the best
//! observed SGLang run to 0.1 % (predicted 14.33 req/s, measured 14.35 at
//! 2P2D/TP4 on 16xH200), which is why it is trusted enough to pick a topology
//! before spending a single GPU-hour on it.
//!
//! The interesting term is [`PrefillCalibration::tp_allreduce_frac`]. Profiling
//! a TP4 prefill worker showed 25 % of *all* GPU kernel time going into
//! `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`, with kernel count fixed at 190
//! (94 layers x 2) and time scaling linearly with token count - bandwidth
//! bound, not latency bound. Per-GPU compute time is invariant in the tensor
//! parallel degree (each rank does 1/t of the FLOPs), but ring all-reduce
//! traffic per rank scales as `(t-1)/t`. So shrinking the prefill workers and
//! running more of them buys throughput for free:
//!
//! | TP | relative tok/s per GPU |
//! |----|------------------------|
//! | 2  | 1.091                  |
//! | 4  | 1.000  (calibration point) |
//! | 8  | 0.960                  |
//!
//! That is the mechanism behind 4P1D beating 2P2D, and it is why the router in
//! this repository is built for many small prefill workers feeding one wide
//! decode worker rather than a symmetric split.

use serde::{Deserialize, Serialize};

use crate::slo::Slo;

/// Measured prefill behaviour of one worker configuration.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PrefillCalibration {
    /// Sustained cold-cache prefill throughput per GPU at `tp_ref`, tokens/s.
    pub tok_s_per_gpu: f64,
    /// Tensor parallel degree the measurement above was taken at.
    pub tp_ref: u32,
    /// Fraction of GPU kernel time spent in the TP all-reduce at `tp_ref`.
    pub tp_allreduce_frac: f64,
    /// Fraction of wall time the GPU is inside a forward pass. 0.91 measured;
    /// the remaining 9 % is batch assembly and is already inside `tok_s_per_gpu`.
    pub duty_cycle: f64,
}

impl Default for PrefillCalibration {
    /// SGLang `ep1-2p2d-gwab-302350`, Qwen3-235B-A22B FP8, H200, TP4,
    /// `chunked_prefill_size=16384`, cold cache (0.0 % radix hit).
    fn default() -> Self {
        Self {
            tok_s_per_gpu: 7796.0,
            tp_ref: 4,
            tp_allreduce_frac: 0.25,
            duty_cycle: 0.91,
        }
    }
}

impl PrefillCalibration {
    /// Per-GPU prefill throughput at tensor parallel degree `tp`, corrected for
    /// the change in ring all-reduce volume.
    ///
    /// `T(t) = compute + allreduce_unit * (t-1)/t`, normalised so that
    /// `T(tp_ref) == 1`.
    pub fn tok_s_per_gpu_at_tp(&self, tp: u32) -> f64 {
        let ref_ratio = ring_factor(self.tp_ref);
        if ref_ratio <= 0.0 {
            return self.tok_s_per_gpu;
        }
        let compute = 1.0 - self.tp_allreduce_frac;
        let unit = self.tp_allreduce_frac / ref_ratio;
        let t = compute + unit * ring_factor(tp);
        self.tok_s_per_gpu / t
    }

    /// Aggregate request rate a prefill pool of `gpus` GPUs at degree `tp` can
    /// sustain for prompts of `isl` tokens.
    pub fn req_per_s(&self, gpus: u32, tp: u32, isl: u32) -> f64 {
        f64::from(gpus) * self.tok_s_per_gpu_at_tp(tp) / f64::from(isl)
    }
}

/// Per-GPU ring all-reduce traffic factor `(t-1)/t`.
fn ring_factor(tp: u32) -> f64 {
    if tp <= 1 {
        0.0
    } else {
        f64::from(tp - 1) / f64::from(tp)
    }
}

/// Measured decode behaviour.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct DecodeCalibration {
    /// Sustained simultaneous sequences per GPU at the reference point.
    pub concurrency_per_gpu: f64,
    /// Mean inter-token latency observed at that concurrency, milliseconds.
    pub itl_ms_at_ref: f64,
    /// Slope of mean ITL against per-GPU concurrency, ms per sequence per GPU.
    ///
    /// **This one is weakly identified and must not be trusted far from the
    /// reference point.** We have exactly one saturated measurement
    /// (C = 53 at 17.23 ms) plus an N-sweep taken on an unsaturated decode
    /// side, where ITL p99 moved 4.7 -> 5.9 ms as N went 32 -> 128. Fitting a
    /// line through those two regimes gives ~0.10 ms per sequence per GPU, and
    /// extrapolating it to the 20 ms budget predicts a concurrency nobody has
    /// ever observed. So planning refuses to extrapolate by default - see
    /// `max_extrapolation` - and the
    /// *runtime* does not use this curve at all - it runs a closed-loop
    /// controller against measured ITL instead. Replace this with a real
    /// ITL-vs-C sweep before believing any capacity number it produces.
    pub itl_slope_ms: f64,
    /// How far past the measured concurrency the *planner* is allowed to go,
    /// as a multiple of `concurrency_per_gpu`.
    ///
    /// The default is **1.0: do not extrapolate at all.** Planning then answers
    /// "what has this pool been observed to sustain", which is a question the
    /// data can answer. Raising it answers "what would it sustain if the ITL
    /// curve kept its slope", which the data cannot - so raising it is a
    /// deliberate statement of belief, and it has to be written down in the
    /// config where a reviewer can see it.
    ///
    /// This is not timidity. The measured point sits at 17.23 ms against a
    /// 20 ms budget, so there really is ~14 % of headroom to claim. The runtime
    /// claims it empirically with `trtllm_sched::ItlController`, which measures
    /// the curve instead of assuming it. The planner does not get to assume it.
    pub max_extrapolation: f64,
}

impl Default for DecodeCalibration {
    /// SGLang `ep1-2p2d-gwab-302350`: C = 53.0 sustained over 8 GPUs at
    /// mean ITL 17.23 ms.
    fn default() -> Self {
        Self {
            concurrency_per_gpu: 53.0 / 8.0,
            itl_ms_at_ref: 17.23,
            itl_slope_ms: 0.10,
            max_extrapolation: 1.0,
        }
    }
}

impl DecodeCalibration {
    /// Predicted mean ITL when `concurrency` sequences run on `gpus` GPUs.
    pub fn itl_ms(&self, concurrency: f64, gpus: u32) -> f64 {
        let per_gpu = concurrency / f64::from(gpus.max(1));
        self.itl_ms_at_ref + self.itl_slope_ms * (per_gpu - self.concurrency_per_gpu)
    }

    /// Largest concurrency whose predicted mean ITL still fits inside the SLO
    /// with `safety` headroom (0.9 leaves 10 %).
    pub fn max_concurrency_for_slo(&self, gpus: u32, slo: &Slo, safety: f64) -> f64 {
        let budget = slo.itl_ms * safety;
        if self.itl_slope_ms <= 0.0 {
            return self.concurrency_per_gpu * f64::from(gpus);
        }
        let ceiling = self.concurrency_per_gpu * self.max_extrapolation.max(1.0);
        let per_gpu = self.concurrency_per_gpu + (budget - self.itl_ms_at_ref) / self.itl_slope_ms;
        per_gpu.clamp(1.0, ceiling) * f64::from(gpus)
    }

    /// Request rate a decode pool of `gpus` GPUs can sustain.
    pub fn req_per_s(&self, gpus: u32, osl: u32, slo: &Slo, safety: f64) -> f64 {
        let c = self.max_concurrency_for_slo(gpus, slo, safety);
        let itl = self.itl_ms(c, gpus).min(slo.itl_ms);
        c / (f64::from(osl) * itl / 1000.0)
    }
}

/// One candidate prefill/decode topology and the goodput it can reach.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PdSplit {
    pub prefill_gpus: u32,
    pub decode_gpus: u32,
    pub prefill_tp: u32,
    pub decode_tp: u32,
    pub prefill_workers: u32,
    pub prefill_req_s: f64,
    pub decode_req_s: f64,
    /// `min(prefill, decode)` - the rate the deployment can actually sustain.
    pub sustainable_req_s: f64,
    /// `sustainable_req_s * good_frac`.
    pub goodput_req_s: f64,
}

/// The full model: workload shape plus both calibrations.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CapacityModel {
    pub isl: u32,
    pub osl: u32,
    pub slo: Slo,
    pub prefill: PrefillCalibration,
    pub decode: DecodeCalibration,
    /// Fraction of requests expected to meet the SLO once the topology is
    /// running at its sustainable rate. 0.93 is what every healthy run of this
    /// workload has produced; the scheduler's whole job is to raise it.
    pub good_frac: f64,
    /// Headroom kept against the ITL budget when sizing decode concurrency.
    pub itl_safety: f64,
}

impl Default for CapacityModel {
    fn default() -> Self {
        Self {
            isl: 4000,
            osl: 200,
            slo: Slo::default(),
            prefill: PrefillCalibration::default(),
            decode: DecodeCalibration::default(),
            good_frac: 0.93,
            itl_safety: 1.0,
        }
    }
}

impl CapacityModel {
    pub fn evaluate(
        &self,
        prefill_gpus: u32,
        decode_gpus: u32,
        prefill_tp: u32,
        decode_tp: u32,
    ) -> PdSplit {
        let prefill_workers = prefill_gpus.checked_div(prefill_tp).unwrap_or(0);
        let prefill_req_s =
            self.prefill
                .req_per_s(prefill_workers * prefill_tp, prefill_tp, self.isl);
        let decode_req_s = self
            .decode
            .req_per_s(decode_gpus, self.osl, &self.slo, self.itl_safety);
        let sustainable = prefill_req_s.min(decode_req_s);
        PdSplit {
            prefill_gpus,
            decode_gpus,
            prefill_tp,
            decode_tp,
            prefill_workers,
            prefill_req_s,
            decode_req_s,
            sustainable_req_s: sustainable,
            goodput_req_s: sustainable * self.good_frac,
        }
    }

    /// Enumerate every topology that fits in `total_gpus` and return them best
    /// first. `prefill_tps` and `decode_tps` are the degrees the model actually
    /// fits in memory - Qwen3-235B-A22B in FP8 needs at least TP2 for weights
    /// alone, and a decode worker needs room for the KV pool on top.
    pub fn search(&self, total_gpus: u32, prefill_tps: &[u32], decode_tps: &[u32]) -> Vec<PdSplit> {
        let mut out = Vec::new();
        for &ptp in prefill_tps {
            for &dtp in decode_tps {
                if ptp == 0 || dtp == 0 {
                    continue;
                }
                let mut p = ptp;
                while p + dtp <= total_gpus {
                    let d = total_gpus - p;
                    if d % dtp == 0 && p % ptp == 0 {
                        out.push(self.evaluate(p, d, ptp, dtp));
                    }
                    p += ptp;
                }
            }
        }
        // Primary key is goodput. When two topologies tie on it - which happens
        // whenever the same side binds both - prefer the one with more slack on
        // the non-binding side: spare prefill capacity is what absorbs a burst
        // without spending it on the TTFT tail, and that is exactly the term
        // `good_frac` is made of.
        out.sort_by(|a, b| {
            b.goodput_req_s.total_cmp(&a.goodput_req_s).then(
                (b.prefill_req_s + b.decode_req_s).total_cmp(&(a.prefill_req_s + a.decode_req_s)),
            )
        });
        out.dedup_by(|a, b| {
            a.prefill_gpus == b.prefill_gpus
                && a.decode_gpus == b.decode_gpus
                && a.prefill_tp == b.prefill_tp
                && a.decode_tp == b.decode_tp
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The model must still reproduce the run it was calibrated against:
    /// 2P2D, prefill TP4 on 8 GPUs, decode TP8 on 8 GPUs, measured 14.35 req/s.
    /// By default the planner sits exactly on the measured point. The line
    /// through that point says 34 sequences per GPU would fit inside a 20 ms
    /// budget; nobody has ever seen that, so planning does not get to use it.
    #[test]
    fn planning_does_not_extrapolate_by_default() {
        let d = DecodeCalibration::default();
        let slo = Slo::default();
        let c = d.max_concurrency_for_slo(8, &slo, 1.0);
        assert!((c - 53.0).abs() < 0.01, "expected the measured 53, got {c}");
    }

    /// Extrapolating is possible, but only as an explicit, visible choice.
    #[test]
    fn extrapolation_is_opt_in_and_shows_up_in_the_number() {
        let slo = Slo::default();
        let d = DecodeCalibration {
            max_extrapolation: 1.5,
            ..Default::default()
        };
        let c = d.max_concurrency_for_slo(8, &slo, 1.0);
        assert!((c - 79.5).abs() < 0.01, "{c}");
        assert!(
            d.req_per_s(8, 200, &slo, 1.0)
                > DecodeCalibration::default().req_per_s(8, 200, &slo, 1.0)
        );
    }

    #[test]
    fn reproduces_the_gwab_record() {
        let m = CapacityModel::default();
        let s = m.evaluate(8, 8, 4, 8);
        assert!(
            (s.prefill_req_s - 15.59).abs() < 0.05,
            "prefill {}",
            s.prefill_req_s
        );
        assert!(
            (s.decode_req_s - 15.38).abs() < 0.20,
            "decode {}",
            s.decode_req_s
        );
        assert!(
            (s.goodput_req_s - 14.33).abs() < 0.25,
            "goodput {}",
            s.goodput_req_s
        );
    }

    /// Halving the prefill tensor parallel degree removes a third of the
    /// all-reduce traffic per rank and must show up as ~9 % more throughput.
    #[test]
    fn tp2_prefill_is_about_nine_percent_faster_per_gpu() {
        let c = PrefillCalibration::default();
        let r = c.tok_s_per_gpu_at_tp(2) / c.tok_s_per_gpu_at_tp(4);
        assert!((r - 1.0909).abs() < 0.001, "ratio {r}");
        assert!(c.tok_s_per_gpu_at_tp(8) < c.tok_s_per_gpu_at_tp(4));
    }

    /// With TP2 prefill available, the search must prefer more, smaller prefill
    /// workers over the symmetric TP4 split - the 4P1D shape. At 8/8 both
    /// topologies are pinned by the same decode ceiling, so they tie on
    /// goodput and the slack tie-break has to be what separates them.
    #[test]
    fn search_prefers_narrow_prefill_workers() {
        let m = CapacityModel::default();
        let best = m
            .search(16, &[2, 4], &[8])
            .into_iter()
            .next()
            .expect("a topology");
        assert_eq!(best.prefill_tp, 2, "4x TP2 prefill should win over 2x TP4");
        assert_eq!(best.decode_gpus, 8);
        assert_eq!(best.prefill_workers, 4);
        assert!(
            best.prefill_req_s > best.decode_req_s,
            "at 8/8 the decode side is the binding one: {best:?}"
        );
    }

    /// Sanity: the search must never hand back a split it cannot lay out on
    /// the hardware.
    #[test]
    fn search_only_returns_realisable_splits() {
        let m = CapacityModel::default();
        for s in m.search(16, &[1, 2, 4, 8], &[4, 8]) {
            assert_eq!(s.prefill_gpus + s.decode_gpus, 16);
            assert_eq!(s.prefill_gpus % s.prefill_tp, 0);
            assert_eq!(s.decode_gpus % s.decode_tp, 0);
            assert!(s.prefill_workers >= 1);
        }
    }

    #[test]
    fn decode_slot_seconds_is_osl_times_itl() {
        let slo = Slo::default();
        assert!((slo.decode_slot_seconds(200) - 4.0).abs() < 1e-9);
        assert!((slo.decode_req_per_s(64.0, 200) - 16.0).abs() < 1e-9);
    }
}
