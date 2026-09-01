//! What to do about the resource that binds, and how strongly it is backed.
//!
//! `CapacityModel::evaluate` names the constraint. Naming it is only half an
//! answer: this session produced four different recommendations whose evidence
//! ranged from "measured on this stack" to "the arithmetic says so", and they
//! were indistinguishable in prose. A reader acted on the weakest one first
//! more than once.
//!
//! So the strength is part of the type. A `Remedy` cannot be constructed
//! without saying where its number came from, and the printer orders by
//! evidence before it orders by size -- a measured 1.3x outranks a derived 2x,
//! because the derived one might be zero.

use serde::{Deserialize, Serialize};

use crate::capacity::{Bottleneck, CapacityModel, MfuBreakdown, PdSplit};

/// How much to believe a remedy's multiplier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Evidence {
    /// Arithmetic from measured quantities, but the combination has never run.
    /// The weakest kind, and the easiest to mistake for the strongest.
    Derived,
    /// An upstream default this deployment overrides, or vice versa, with no
    /// measurement on either side.
    Untested,
    /// Measured on a different stack, where the mechanism is general enough to
    /// transfer -- an all-to-all removed is an all-to-all removed.
    Transferred,
    /// Measured on this stack, this model, this hardware.
    Measured,
}

impl Evidence {
    pub fn label(self) -> &'static str {
        match self {
            Evidence::Measured => "measured",
            Evidence::Transferred => "transferred",
            Evidence::Untested => "untested",
            Evidence::Derived => "derived",
        }
    }
}

/// One knob, what to set it to, what it is expected to buy, and why.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Remedy {
    pub knob: &'static str,
    pub setting: &'static str,
    /// Throughput multiplier if it works. 1.0 means "unblocks something" rather
    /// than "goes faster".
    pub multiplier: f64,
    pub evidence: Evidence,
    pub because: &'static str,
    /// Extra variables this remedy needs alongside its own, or "" for none.
    /// Context parallelism changes the GPUs a worker owns, so it cannot be a
    /// one-variable change however much a table wants it to be.
    pub with: &'static str,
}

impl Remedy {
    /// The line to run. A recommendation a reader has to translate into a
    /// command is a recommendation they will translate wrong.
    pub fn command(&self, script: &str) -> String {
        let mut vars = format!("{}={}", self.knob, self.setting);
        if !self.with.is_empty() {
            vars.push(',');
            vars.push_str(self.with);
        }
        // One line, because a wrapped command is a command someone will paste
        // half of. 25a-hgpn143's NCCL hangs on its first collective, so the
        // exclusion is not optional and belongs in the line rather than in a
        // note beside it.
        format!(
            "{} {} {}",
            "sbatch -p 16gpus -N 2 --gres=gpu:H200:8 -t 00:15:00",
            "--exclude=25a-hgpn142,25a-hgpn143",
            format_args!("--export=ALL,{vars} {script}")
        )
    }
}

impl CapacityModel {
    /// Remedies for `split`, strongest evidence first and largest first within
    /// a level.
    pub fn remedies(&self, split: &PdSplit) -> Vec<Remedy> {
        let b = self.prefill.mfu_breakdown();
        let mut out = match split.bottleneck {
            Bottleneck::KvTransfer => Self::transfer_remedies(),
            Bottleneck::Prefill => Self::prefill_remedies(&b),
            Bottleneck::Decode => Self::decode_remedies(),
        };
        // Drop anything this split already does. A remedy that restates the
        // current configuration is noise, and noise in a list ordered by
        // evidence is worse than noise anywhere else -- it occupies the slot a
        // reader trusts most.
        out.retain(|r| !self.already_applied(r, split));
        out.sort_by(|a, b| {
            b.evidence
                .cmp(&a.evidence)
                .then(b.multiplier.total_cmp(&a.multiplier))
        });
        out
    }

    /// Whether `split` already reflects this remedy.
    ///
    /// Only the knobs the model can see. The rest live in the launcher's
    /// environment, which this crate deliberately does not read -- a model
    /// that parses the deployment to describe it would be describing itself.
    fn already_applied(&self, r: &Remedy, split: &PdSplit) -> bool {
        match r.knob {
            "DECODE_TP" => split.decode_tp == 4,
            // Speculation is in the decode calibration, not the launcher.
            "SPEC_DECODE" => self.decode.speculation.is_some(),
            "KV_XFER_CONCURRENCY" => self.xfer_concurrency >= 16,
            _ => false,
        }
    }

    fn transfer_remedies() -> Vec<Remedy> {
        vec![Remedy {
            knob: "KV_XFER_CONCURRENCY",
            setting: "16",
            multiplier: 16.0,
            evidence: Evidence::Derived,
            because: "upstream serialises the handoff to one buffer in each \
                      direction (baseTransBuffer.cpp:109); the pool size is the \
                      concurrency",
            with: "",
        }]
    }

    fn prefill_remedies(b: &MfuBreakdown) -> Vec<Remedy> {
        vec![
            Remedy {
                knob: "EXPERT_PARALLEL",
                setting: "1",
                multiplier: 1.43,
                evidence: Evidence::Transferred,
                because: "SGLang job 302350 measured EP1 at +43% over EP4 on \
                          this model and hardware; removing a per-token \
                          all-to-all transfers between stacks",
                with: "",
            },
            Remedy {
                knob: "TORCH_COMPILE",
                setting: "inductor",
                multiplier: b.compute_mfu_worth(0.50),
                evidence: Evidence::Derived,
                because: "the compute phase runs at 25.3% MFU and inductor is \
                          the codegen that would change the GEMMs",
                with: "",
            },
            Remedy {
                knob: "MOE_BACKEND",
                setting: "DEEPGEMM",
                multiplier: 19.8 / 17.0,
                evidence: Evidence::Transferred,
                because: "AUTO resolves to CUTLASS on SM90. deep_gemm plus \
                          expert parallelism off took a neighbouring stack from \
                          goodput 17.0 to 19.8 on this model and hardware -- it \
                          was listed here as untested until that run existed",
                with: "",
            },
            Remedy {
                knob: "KV_CACHE_DTYPE",
                setting: "fp8",
                multiplier: 1.0,
                evidence: Evidence::Transferred,
                because: "halves resident KV, which is what makes TP2 prefill \
                          fit at all: 20.9 GiB per rank holds max_num_tokens of \
                          in-flight KV in fp8 and not in fp16, and our own three \
                          TP2 failures were all fp16",
                with: "",
            },
            Remedy {
                knob: "PREFILL_MAX_NUM_TOKENS",
                setting: "16384",
                multiplier: 1.0,
                evidence: Evidence::Transferred,
                because: "four whole ISL-4000 prompts per iteration against \
                          upstream's 8192; this is also the quantity our own two \
                          models disagree about, so measuring it settles the \
                          14.30-versus-7.53 gap as a side effect",
                with: "",
            },
            Remedy {
                knob: "ALLREDUCE_STRATEGY",
                setting: "LOWPRECISION",
                multiplier: b.allreduce_worth,
                evidence: Evidence::Derived,
                because: "the TP all-reduce is 25% of prefill kernel time",
                with: "",
            },
            Remedy {
                knob: "HOST_DISPATCH_SPIN",
                setting: "1",
                multiplier: b.duty_cycle_worth,
                evidence: Evidence::Derived,
                because: "9% of wall time is outside the forward pass; the \
                          stated cost is one spinning core and this node has 96",
                with: "",
            },
            Remedy {
                knob: "CONTEXT_PARALLEL",
                setting: "2",
                multiplier: 1.0,
                evidence: Evidence::Derived,
                because: "splits the sequence rather than the weights, so it \
                          lowers prefill LATENCY -- which is what the TTFT gate \
                          is written in -- rather than throughput",
                with: "PREFILL_WORKERS=1,PREFILL_TP=4",
            },
        ]
    }

    fn decode_remedies() -> Vec<Remedy> {
        vec![
            Remedy {
                knob: "SPEC_DECODE",
                setting: "1",
                multiplier: 19.18 / 11.08,
                evidence: Evidence::Transferred,
                because: "EAGLE3 at one draft token measured ITL p95 11.08 ms \
                          against 19.18 baseline on this model and this \
                          hardware, with acceptance 1.82 and 12/12 outputs \
                          token-identical under greedy sampling; topk=2 cost 5% \
                          of goodput and topk=4 collapsed to 3.29",
                with: "",
            },
            Remedy {
                knob: "DECODE_TP",
                setting: "4",
                multiplier: 2.7,
                evidence: Evidence::Measured,
                because: "one TP8 decode worker delivered 815 tok/s where two \
                          TP4 workers reach 2170-2470; Qwen3-235B has 4 KV \
                          heads, so TP8 must duplicate them",
                with: "",
            },
            Remedy {
                knob: "CUDA_GRAPH_MAX_BATCH",
                setting: "96",
                multiplier: 1.0,
                evidence: Evidence::Untested,
                because: "CudaGraphConfig defaults max_batch_size to 0, so a \
                          capture that is merely enabled captures nothing",
                with: "",
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> CapacityModel {
        CapacityModel::default()
    }

    /// Evidence outranks size. A measured 1.3x is worth acting on before a
    /// derived 2x, because the derived one might turn out to be 1.0.
    #[test]
    fn remedies_are_ordered_by_how_well_they_are_backed() {
        let m = model();
        let split = m.evaluate(8, 8, 4, 4);
        let r = m.remedies(&split);
        assert!(!r.is_empty(), "no remedy for {:?}", split.bottleneck);
        for w in r.windows(2) {
            assert!(
                w[0].evidence >= w[1].evidence,
                "{} ({}) came before {} ({})",
                w[0].knob,
                w[0].evidence.label(),
                w[1].knob,
                w[1].evidence.label()
            );
        }
    }

    /// Every bottleneck the model can report must have something to say about
    /// it. A named constraint with no remedy is a diagnosis with no treatment.
    #[test]
    fn every_bottleneck_has_at_least_one_remedy() {
        let m = model();
        for (p, d, ptp, dtp) in [(8, 8, 4, 4), (12, 4, 4, 4), (4, 12, 4, 4)] {
            let split = m.evaluate(p, d, ptp, dtp);
            assert!(
                !m.remedies(&split).is_empty(),
                "{:?} at {p}/{d} has no remedy",
                split.bottleneck
            );
        }
        // And the transfer case, which needs the serialised configuration.
        let serial = CapacityModel {
            xfer_concurrency: 1,
            ..CapacityModel::default()
        };
        let split = serial.evaluate(8, 8, 4, 8);
        assert_eq!(split.bottleneck, Bottleneck::KvTransfer);
        assert!(!serial.remedies(&split).is_empty());
    }

    /// The multipliers come from the MFU decomposition, not from prose, so a
    /// recalibration moves them.
    #[test]
    fn prefill_multipliers_track_the_mfu_breakdown() {
        let m = model();
        let b = m.prefill.mfu_breakdown();
        // Few prefill GPUs against many decode: prefill binds.
        let split = m.evaluate(4, 12, 4, 4);
        assert_eq!(split.bottleneck, Bottleneck::Prefill);
        let r = m.remedies(&split);

        let allreduce = r
            .iter()
            .find(|x| x.knob == "ALLREDUCE_STRATEGY")
            .expect("no all-reduce remedy");
        assert!((allreduce.multiplier - b.allreduce_worth).abs() < 1e-9);

        let compile = r
            .iter()
            .find(|x| x.knob == "TORCH_COMPILE")
            .expect("no compile remedy");
        assert!((compile.multiplier - b.compute_mfu_worth(0.50)).abs() < 1e-9);
        assert!(
            compile.multiplier > allreduce.multiplier,
            "the grouped GEMM should still outrank the collective: {:.2} vs {:.2}",
            compile.multiplier,
            allreduce.multiplier
        );
    }

    /// A recommendation a reader has to translate into a command is one they
    /// will translate wrong -- and CONTEXT_PARALLEL is the proof, because it
    /// changes how many GPUs a worker owns and is not a one-variable change
    /// however much a table wants it to be.
    #[test]
    fn a_remedy_emits_a_command_that_carries_its_dependencies() {
        let m = model();
        let split = m.evaluate(4, 12, 4, 4);
        let r = m.remedies(&split);

        let cp = r
            .iter()
            .find(|x| x.knob == "CONTEXT_PARALLEL")
            .expect("no context-parallel remedy");
        let cmd = cp.command("scripts/stage-d-235b-disagg.sbatch");
        assert!(cmd.contains("CONTEXT_PARALLEL=2"), "{cmd}");
        assert!(
            cmd.contains("PREFILL_WORKERS=1") && cmd.contains("PREFILL_TP=4"),
            "context parallelism changes the worker's GPU count and the command \
             did not say so: {cmd}"
        );

        // Every setting must be a value a shell can take, not a description.
        for x in &r {
            assert!(
                !x.setting.contains(' '),
                "{}={} is prose, not a value",
                x.knob,
                x.setting
            );
        }
    }

    /// A remedy that restates the current configuration is noise, and it lands
    /// in the slot a reader trusts most -- top of a list ordered by evidence.
    #[test]
    fn a_remedy_already_applied_is_not_offered() {
        let m = model();
        let split = m.evaluate(12, 4, 4, 4);
        assert_eq!(split.decode_tp, 4, "this test needs TP4 decode");
        assert!(
            !m.remedies(&split).iter().any(|r| r.knob == "DECODE_TP"),
            "DECODE_TP=4 was offered to a split that already runs TP4"
        );

        // At TP8 it is the strongest thing the model can say, so it must appear.
        let split8 = m.evaluate(12, 8, 4, 8);
        let r8 = m.remedies(&split8);
        if split8.bottleneck == Bottleneck::Decode {
            assert!(
                r8.iter().any(|r| r.knob == "DECODE_TP"),
                "a TP8 decode split was told nothing about TP8"
            );
        }
    }

    /// Likewise for the transfer: sixteen buffers means the serialisation
    /// remedy has been taken.
    #[test]
    fn the_transfer_remedy_disappears_once_it_is_configured() {
        let serial = CapacityModel {
            xfer_concurrency: 1,
            ..CapacityModel::default()
        };
        let split = serial.evaluate(8, 8, 4, 8);
        assert!(serial
            .remedies(&split)
            .iter()
            .any(|r| r.knob == "KV_XFER_CONCURRENCY"));

        let m = model();
        assert!(m.xfer_concurrency >= 16);
        let split = m.evaluate(8, 8, 4, 8);
        assert!(!m
            .remedies(&split)
            .iter()
            .any(|r| r.knob == "KV_XFER_CONCURRENCY"));
    }

    /// The one measured remedy must be labelled as such, because it is the
    /// only one a reader should act on without a preliminary run.
    #[test]
    fn the_tp8_decode_finding_is_labelled_measured() {
        let m = model();
        // Many prefill GPUs against few decode: decode binds.
        let split = m.evaluate(12, 4, 4, 4);
        assert_eq!(split.bottleneck, Bottleneck::Decode);
        let r = m.remedies(&split);
        let tp = r.iter().find(|x| x.knob == "DECODE_TP");
        if let Some(tp) = tp {
            assert_eq!(tp.evidence, Evidence::Measured);
        }
    }
}
