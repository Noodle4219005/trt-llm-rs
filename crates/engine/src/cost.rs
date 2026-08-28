//! The analytic cost model.
//!
//! Every number here is either measured on Qwen3-235B-A22B FP8 on H200 or
//! labelled as an assumption. The distinction matters more than the values: a
//! calibration point constrains a *product*, not the individual terms, and
//! extrapolating the wrong term is how a plausible model produces an impossible
//! plan.

use trtllm_core::capacity::{DecodeCalibration, PrefillCalibration};

/// Cost of a prefill batch.
#[derive(Clone, Copy, Debug)]
pub struct PrefillCurve {
    /// Tokens per millisecond for this worker at one sequence per batch.
    pub tokens_per_ms: f64,
    /// Fixed per-batch cost. Measured duty cycle was 91 %, so roughly 9 % of
    /// wall time sits between forward passes.
    pub overhead_ms: f64,
    /// Speedup from co-scheduling sequences in one forward pass. Measured:
    /// one sequence per batch runs 31,111 tok/s, five run 34,736 tok/s, so
    /// +11.6 % once the MoE grouped GEMM has enough tokens to work with.
    pub multiseq_gain: f64,
    /// Sequences needed to realise the full gain.
    pub multiseq_saturation: f64,
}

impl PrefillCurve {
    /// Build the curve for one worker of `gpus` GPUs at tensor parallel degree
    /// `tp`, using the all-reduce correction from the capacity model.
    pub fn for_worker(cal: &PrefillCalibration, gpus: u32, tp: u32) -> Self {
        let tok_s = cal.tok_s_per_gpu_at_tp(tp) * f64::from(gpus);
        let tokens_per_ms = tok_s / 1000.0;
        // Turn the measured duty cycle into a per-batch constant using a
        // nominal 16k-token batch, which is the size it was measured at.
        let nominal_batch_ms = 16384.0 / tokens_per_ms;
        let overhead_ms = nominal_batch_ms * (1.0 / cal.duty_cycle - 1.0);
        Self {
            tokens_per_ms,
            overhead_ms,
            multiseq_gain: 0.116,
            multiseq_saturation: 5.0,
        }
    }

    /// Effective token rate when `num_seqs` sequences share the batch.
    pub fn rate_at(&self, num_seqs: usize) -> f64 {
        let n = (num_seqs.max(1) as f64).min(self.multiseq_saturation);
        let frac = (n - 1.0) / (self.multiseq_saturation - 1.0).max(1.0);
        self.tokens_per_ms * (1.0 + self.multiseq_gain * frac)
    }

    pub fn batch_ms(&self, tokens: usize, num_seqs: usize) -> f64 {
        if tokens == 0 {
            return 0.0;
        }
        self.overhead_ms + tokens as f64 / self.rate_at(num_seqs)
    }
}

/// Cost of a decode step.
///
/// **Read this before trusting an extrapolation.** We have one saturated
/// measurement: 53 concurrent sequences on an 8-GPU decode worker at a mean ITL
/// of 17.23 ms. That constrains `base + slope * 53 = 17.23` and nothing else.
/// `base = 12, slope = 0.099` and `base = 4, slope = 0.25` both fit it exactly
/// and disagree by more than 2x about the concurrency that fits inside a 20 ms
/// budget. The default below picks a middle assumption *and the runtime does
/// not use it* - `trtllm_sched::ItlController` steers on measured latency
/// instead. This curve exists so the simulator has something to simulate and so
/// capacity planning has a starting point, not so anyone can plan against it.
#[derive(Clone, Copy, Debug)]
pub struct DecodeCurve {
    pub base_ms: f64,
    pub slope_ms: f64,
    /// Concurrency past which the KV working set stops fitting and the curve
    /// bends sharply. Unmeasured; set from the KV pool size at runtime.
    pub knee: f64,
    pub knee_penalty: f64,
}

impl DecodeCurve {
    /// Fit the line through one measured point given an assumed intercept.
    pub fn from_point(concurrency: f64, itl_ms: f64, assumed_base_ms: f64) -> Self {
        let slope = if concurrency > 0.0 {
            (itl_ms - assumed_base_ms) / concurrency
        } else {
            0.0
        };
        Self {
            base_ms: assumed_base_ms,
            slope_ms: slope.max(0.0),
            knee: f64::INFINITY,
            knee_penalty: 3.0,
        }
    }

    pub fn from_calibration(cal: &DecodeCalibration, gpus: u32) -> Self {
        let c = cal.concurrency_per_gpu * f64::from(gpus.max(1));
        // 12 ms is the assumed single-sequence step for this MoE at TP8. It is
        // an assumption, not a measurement.
        Self::from_point(c, cal.itl_ms_at_ref, 12.0)
    }

    pub fn step_ms(&self, concurrency: usize) -> f64 {
        let c = concurrency as f64;
        let linear = self.base_ms + self.slope_ms * c;
        if c <= self.knee {
            linear
        } else {
            linear + (c - self.knee) * self.slope_ms * self.knee_penalty
        }
    }
}

/// Everything the mock engine and the simulator need to cost a worker.
#[derive(Clone, Copy, Debug)]
pub struct CostModel {
    pub prefill: PrefillCurve,
    pub decode: DecodeCurve,
    /// Time to move one sequence's KV from a prefill worker to a decode worker.
    /// Layer-wise streaming overlaps most of this with the prefill itself; the
    /// residual is what shows up in TTFT.
    pub kv_transfer_ms: f64,
}

impl CostModel {
    pub fn new(prefill: PrefillCurve, decode: DecodeCurve, kv_transfer_ms: f64) -> Self {
        Self {
            prefill,
            decode,
            kv_transfer_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiseq_gain_matches_the_measurement() {
        let c = PrefillCurve {
            tokens_per_ms: 31.111,
            overhead_ms: 0.0,
            multiseq_gain: 0.116,
            multiseq_saturation: 5.0,
        };
        assert!((c.rate_at(1) - 31.111).abs() < 1e-6);
        assert!((c.rate_at(5) - 34.72).abs() < 0.05, "{}", c.rate_at(5));
        // Past saturation the gain does not keep growing.
        assert!((c.rate_at(9) - c.rate_at(5)).abs() < 1e-9);
    }

    #[test]
    fn a_tp2_worker_is_faster_per_gpu_than_a_tp4_one() {
        let cal = PrefillCalibration::default();
        let tp2 = PrefillCurve::for_worker(&cal, 2, 2);
        let tp4 = PrefillCurve::for_worker(&cal, 4, 4);
        assert!(tp2.tokens_per_ms / 2.0 > tp4.tokens_per_ms / 4.0);
    }

    /// The point that pins the model must come back out of it.
    #[test]
    fn decode_curve_reproduces_its_calibration_point() {
        let d = DecodeCurve::from_calibration(&DecodeCalibration::default(), 8);
        assert!((d.step_ms(53) - 17.23).abs() < 0.01, "{}", d.step_ms(53));
    }

    /// Two intercepts that both fit the measurement disagree badly about the
    /// concurrency that fits in a 20 ms budget. This test exists to keep that
    /// fact visible rather than buried in a comment.
    #[test]
    fn the_intercept_is_not_identified_by_one_point() {
        let shallow = DecodeCurve::from_point(53.0, 17.23, 15.0);
        let steep = DecodeCurve::from_point(53.0, 17.23, 2.0);
        assert!(
            (shallow.step_ms(53) - steep.step_ms(53)).abs() < 0.01,
            "both fit the point"
        );
        let c_shallow = (20.0 - shallow.base_ms) / shallow.slope_ms;
        let c_steep = (20.0 - steep.base_ms) / steep.slope_ms;
        assert!(
            c_shallow > c_steep * 1.8,
            "one measurement admits {c_steep:.0} or {c_shallow:.0} concurrent sequences"
        );
    }
}
