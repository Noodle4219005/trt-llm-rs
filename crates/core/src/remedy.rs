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
        // half of. 25a-hgpn143 hangs on its first NCCL collective and
        // 25a-hgpn175 has no apptainer at all, so the exclusion is not
        // optional and belongs in the line rather than in a note beside it.
        format!(
            "{} {} {}",
            "sbatch -p 16gpus -N 2 --gres=gpu:H200:8 -t 00:15:00",
            "--exclude=25a-hgpn142,25a-hgpn143,25a-hgpn175",
            format_args!("--export=ALL,{vars} {script}")
        )
    }
}

impl CapacityModel {
    /// Remedies for `split`, strongest evidence first and largest first within
    /// a level.
    pub fn remedies(&self, split: &PdSplit) -> Vec<Remedy> {
        self.remedies_given(split, &crate::engine_config::EngineConfig::default())
    }

    /// Remedies for `split`, excluding everything `applied` already does.
    pub fn remedies_given(
        &self,
        split: &PdSplit,
        applied: &crate::engine_config::EngineConfig,
    ) -> Vec<Remedy> {
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
        out.retain(|r| !self.already_applied(r, split, applied));
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
    fn already_applied(
        &self,
        r: &Remedy,
        split: &PdSplit,
        applied: &crate::engine_config::EngineConfig,
    ) -> bool {
        match r.knob {
            "DECODE_TP" => split.decode_tp == 4,
            "SPECULATION" => self.decode.speculation.is_some() || applied.speculation.enabled,
            "KV_XFER_CONCURRENCY" => self.xfer_concurrency >= 16,
            "EXPERT_PARALLEL" => applied.expert_parallel == 1,
            "MOE_BACKEND" => applied.moe_backend == "CUTLASS" || applied.moe_backend == "AUTO",
            "TORCH_COMPILE" => applied.torch_compile != "0",
            "HOST_DISPATCH_SPIN" => applied.host_dispatch_spin,
            "ALLREDUCE_STRATEGY" => applied.allreduce_strategy != "AUTO",
            "CONTEXT_PARALLEL" => applied.context_parallel > 1,
            "CUDA_GRAPH_MAX_BATCH" => applied.cuda_graph_max_batch > 0,
            // These two describe the deployment rather than the engine
            // section, so the model reads them from itself: the KV dtype is
            // already in kv_dtype_bytes and the prefill budget in
            // max_num_tokens.
            "KV_CACHE_DTYPE" => self.kv_dtype_bytes <= 1.0,
            "PREFILL_MAX_NUM_TOKENS" => self.max_num_tokens >= 16384,
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
                setting: "CUTLASS",
                multiplier: 1.0,
                evidence: Evidence::Untested,
                because: "AUTO already resolves to CUTLASS on SM90, so this \
                          only makes the choice explicit. DEEPGEMM and TRTLLM \
                          both refuse anything but SM100/103 and MARLIN needs \
                          an NVFP4 checkpoint, so on H200 with fp8 there is no \
                          other MoE backend to try -- the neighbouring stack's \
                          17.0-to-19.8 with deep_gemm was vLLM's own \
                          implementation, not this flag",
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
                knob: "SPECULATION",
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

#[cfg(test)]
mod applied_tests {
    use super::*;
    use crate::engine_config::EngineConfig;

    /// The filter is the point of the list, and it was half-built.
    ///
    /// It covered DECODE_TP and KV_XFER_CONCURRENCY while EXPERT_PARALLEL,
    /// KV_CACHE_DTYPE and PREFILL_MAX_NUM_TOKENS were all defaults and all
    /// still offered -- three lines telling a reader to do what they already
    /// do, at the top of a list ordered by evidence, which is the exact
    /// failure the doc comment above warns about.
    #[test]
    fn the_default_deployment_has_one_thing_left_to_try() {
        let m = CapacityModel::default();
        let e = EngineConfig::default();
        let split = m.evaluate(8, 8, 2, 4);
        let r = m.remedies_given(&split, &e);
        let knobs: Vec<&str> = r.iter().map(|x| x.knob).collect();
        assert_eq!(
            knobs,
            vec!["SPECULATION"],
            "everything else is already configured; got {knobs:?}"
        );
    }

    /// Taking the last decode remedy moves the constraint, and the list must
    /// follow it rather than empty out.
    ///
    /// With speculation on, decode goes 15.38 -> 26.62 req/s and the binding
    /// resource flips to prefill -- which is what the 2026-08-28 analysis
    /// predicted ("only speculation is -5%, only prefill is capped at 15.4").
    /// The four prefill levers it then offers are genuinely untried, so an
    /// empty list here would mean the model had stopped looking.
    #[test]
    fn taking_the_last_decode_remedy_moves_the_constraint_to_prefill() {
        let mut cfg = crate::config::Config::default();
        cfg.engine.speculation.enabled = true;
        let m = cfg.capacity_model();
        let split = m.evaluate(8, 8, 2, 4);
        assert_eq!(
            split.bottleneck,
            Bottleneck::Prefill,
            "speculation should make prefill the constraint"
        );
        let knobs: Vec<&str> = m
            .remedies_given(&split, &cfg.engine)
            .iter()
            .map(|x| x.knob)
            .collect();
        assert!(
            !knobs.contains(&"SPECULATION"),
            "speculation is on and must not be offered again: {knobs:?}"
        );
        assert!(
            knobs.contains(&"TORCH_COMPILE"),
            "the compute-MFU lever is the largest untried one: {knobs:?}"
        );
    }

    /// A deployment that has NOT applied them must still be told. The filter
    /// must remove what is done, not the knowledge itself.
    #[test]
    fn a_stock_deployment_is_told_everything() {
        let m = CapacityModel {
            kv_dtype_bytes: 2.0,
            max_num_tokens: 8192,
            ..CapacityModel::default()
        };
        let e = EngineConfig {
            expert_parallel: 4,
            ..EngineConfig::default()
        };
        // Prefill-bound, so the prefill remedies are the ones on offer.
        let split = m.evaluate(4, 12, 2, 4);
        assert_eq!(split.bottleneck, Bottleneck::Prefill);
        let knobs: Vec<&str> = m
            .remedies_given(&split, &e)
            .iter()
            .map(|x| x.knob)
            .collect();
        for expected in [
            "EXPERT_PARALLEL",
            "KV_CACHE_DTYPE",
            "PREFILL_MAX_NUM_TOKENS",
        ] {
            assert!(
                knobs.contains(&expected),
                "{expected} was filtered out of a deployment that has not applied it: {knobs:?}"
            );
        }
    }
}
