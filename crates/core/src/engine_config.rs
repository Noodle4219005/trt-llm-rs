//! Engine knobs a user tunes, and the launcher environment they become.
//!
//! There were two parameter systems: 28 shell variables in the launcher and 50
//! fields in this config, overlapping partly and agreeing by hand. A user
//! tuning the TOML did not change the engine.yaml the launcher wrote, and a
//! test checked six of the twenty-eight.
//!
//! This is the single source. `Config::to_env` emits every launcher variable,
//! the launcher sources it, and a test asserts that every `: "${X:=` in the
//! script is emitted here -- so a knob added to one side and not the other is a
//! test failure rather than a silent divergence discovered on a GPU.

use serde::{Deserialize, Serialize};

/// How the engine executes, as opposed to how big the deployment is.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct EngineConfig {
    /// Attention kernel: TRTLLM, FLASHINFER, TORCH, VANILLA.
    pub attn_backend: String,
    /// MoE kernel: AUTO, CUTLASS, DEEPGEMM, TRTLLM, TRITON, VANILLA.
    ///
    /// AUTO resolves to CUTLASS on SM90. DEEPGEMM plus expert parallelism off
    /// took a neighbouring stack from goodput 17.0 to 19.8 on this model.
    pub moe_backend: String,
    /// TP all-reduce: AUTO, NCCL, ONESHOT, TWOSHOT, LOWPRECISION, MNNVL, UB,
    /// NCCL_SYMMETRIC, AUTO_LOWPRECISION. 25% of prefill kernel time.
    pub allreduce_strategy: String,
    /// Expert parallel degree, or 1 for none. Upstream's own Mapping defaults
    /// to 1; EP4 measured -43% against EP1 on this model.
    pub expert_parallel: u32,
    /// Attention data parallelism on the prefill side.
    pub prefill_attention_dp: bool,
    /// Chunked prefill. Off: it splits a prompt across iterations, which raises
    /// TTFT for a workload whose prompts already fit.
    pub chunked_prefill: bool,
    /// CUDA graph capture size, before speculation multiplies it.
    ///
    /// CudaGraphConfig defaults max_batch_size to 0, so a capture that is
    /// merely enabled captures nothing.
    pub cuda_graph_max_batch: u32,
    /// torch.compile: "0" off, "1" upstream defaults, "inductor" with the
    /// codegen aimed at the 25.3% compute MFU. Compilation time is charged to
    /// the first requests' TTFT.
    pub torch_compile: String,
    /// Spin-wait CUDA host dispatch. Costs one spinning core, of 96.
    pub host_dispatch_spin: bool,
    /// KV block reuse -- TRT-LLM's prefix cache.
    ///
    /// Defaults to TRUE upstream and this deployment sets it false. A
    /// teammate's 17.67 was withdrawn on audit for exactly this: a 99.1% hit
    /// rate, prefill doing 1% of the work the benchmark meant to measure.
    /// Turn it on for production serving; leave it off for any number that has
    /// to survive an audit.
    pub block_reuse: bool,
    /// Speculative decoding on the decode side.
    pub speculation: SpeculationConfig,
    /// Sequence parallelism over the context. Splits the sequence rather than
    /// the weights, so it lowers prefill latency rather than raising
    /// throughput -- the quantity the TTFT gate is written in.
    pub context_parallel: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct SpeculationConfig {
    /// Off by default. EAGLE3 at one draft token measured ITL p95 11.08 ms
    /// against 19.18 on this model and hardware, but on a different serving
    /// stack.
    pub enabled: bool,
    /// Draft checkpoint. LlamaForCausalLMEagle3 is what Eagle3DecodingConfig
    /// expects.
    pub model: String,
    /// Draft tokens per step. topk=2 cost 5% of goodput and topk=4 collapsed.
    pub draft_tokens: u32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            attn_backend: "TRTLLM".into(),
            moe_backend: "AUTO".into(),
            allreduce_strategy: "AUTO".into(),
            expert_parallel: 1,
            prefill_attention_dp: false,
            chunked_prefill: false,
            cuda_graph_max_batch: 96,
            torch_compile: "0".into(),
            host_dispatch_spin: false,
            block_reuse: false,
            speculation: SpeculationConfig::default(),
            context_parallel: 1,
        }
    }
}

impl Default for SpeculationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: "/work/u4063814/hf_cache/eagle3/qwen3-235b-instruct-2507-specforge".into(),
            draft_tokens: 1,
        }
    }
}

/// Allowed values, so a typo is caught here rather than by a worker that
/// starts, validates, and exits during a GPU allocation.
impl EngineConfig {
    pub const ATTN_BACKENDS: &'static [&'static str] =
        &["TRTLLM", "FLASHINFER", "TORCH", "VANILLA"];
    /// Exactly the Literal in MoeConfig.backend. This list says what TRT-LLM
    /// accepts as a name; `validate_for` says what this machine can run. Two
    /// different questions, and collapsing them is what let DEEPGEMM be
    /// recommended for an H200.
    pub const MOE_BACKENDS: &'static [&'static str] = &[
        "AUTO",
        "CUTLASS",
        "CUTEDSL",
        "WIDEEP",
        "TRTLLM",
        "DEEPGEMM",
        "DENSEGEMM",
        "VANILLA",
        "TRITON",
        "MARLIN",
        "MEGAMOE_DEEPGEMM",
    ];
    pub const ALLREDUCE_STRATEGIES: &'static [&'static str] = &[
        "AUTO",
        "NCCL",
        "ONESHOT",
        "TWOSHOT",
        "LOWPRECISION",
        "MNNVL",
        "UB",
        "NCCL_SYMMETRIC",
        "AUTO_LOWPRECISION",
    ];
    pub const TORCH_COMPILE: &'static [&'static str] = &["0", "1", "inductor"];

    pub fn validate(&self) -> Result<(), String> {
        let one_of = |name: &str, value: &str, allowed: &[&str]| -> Result<(), String> {
            if allowed.contains(&value) {
                Ok(())
            } else {
                Err(format!(
                    "{name} = \"{value}\" is not one of {}",
                    allowed.join(", ")
                ))
            }
        };
        one_of("attn_backend", &self.attn_backend, Self::ATTN_BACKENDS)?;
        one_of("moe_backend", &self.moe_backend, Self::MOE_BACKENDS)?;
        one_of(
            "allreduce_strategy",
            &self.allreduce_strategy,
            Self::ALLREDUCE_STRATEGIES,
        )?;
        one_of("torch_compile", &self.torch_compile, Self::TORCH_COMPILE)?;
        if self.cuda_graph_max_batch == 0 {
            return Err("cuda_graph_max_batch = 0 captures nothing, which is \
                        indistinguishable from having no graph at all. Set it \
                        at or above the decode residency, or say so by leaving \
                        it at the default."
                .into());
        }
        if self.context_parallel > 1 && self.speculation.enabled {
            return Err("context_parallel and speculation have not been run \
                        together on this stack. Enable one."
                .into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A misspelling must be caught here. The alternative is a worker that
    /// starts, allocates, validates and exits inside a GPU allocation, which
    /// is the most expensive place to learn about a typo -- and this
    /// deployment has paid for that lesson twice.
    #[test]
    fn a_misspelt_backend_is_rejected_with_the_alternatives() {
        let c = EngineConfig {
            moe_backend: "DEEP_GEMM".into(),
            ..EngineConfig::default()
        };
        let e = c.validate().expect_err("DEEP_GEMM should not validate");
        assert!(
            e.contains("DEEPGEMM"),
            "the error must list what is valid: {e}"
        );

        let c = EngineConfig {
            attn_backend: "flashinfer".into(),
            ..EngineConfig::default()
        };
        assert!(
            c.validate().is_err(),
            "backend names are upper case and the check is exact; a lower-case \
             one that silently fell through would be found on a GPU"
        );
    }

    /// Every default must validate, or the first thing a user does is see an
    /// error about a file they have not touched.
    #[test]
    fn the_defaults_validate() {
        EngineConfig::default().validate().expect("defaults");
    }

    /// Zero is the value CudaGraphConfig uses to mean "capture nothing", and a
    /// user who writes it almost certainly meant to disable the graph rather
    /// than to configure an empty one.
    #[test]
    fn a_zero_graph_capture_is_rejected_as_the_mistake_it_usually_is() {
        let c = EngineConfig {
            cuda_graph_max_batch: 0,
            ..EngineConfig::default()
        };
        let e = c.validate().expect_err("0 should not validate");
        assert!(e.contains("captures nothing"), "{e}");
    }

    /// Speculation multiplies the batch dimension; the capture has to include
    /// it or every speculative step misses the graph.
    #[test]
    fn the_graph_must_be_captured_at_the_speculative_batch_size() {
        let s = crate::capacity::Speculation::eagle3_topk1();
        assert_eq!(s.graph_batch_for(96), 192);
        assert_eq!(s.graph_batch_for(64), 128);
    }

    /// Untried combinations are refused rather than silently attempted. This
    /// one is refused because nothing has run it, which is a different claim
    /// from it being wrong -- the message says so.
    #[test]
    fn an_uncharted_combination_is_refused_and_says_why() {
        let c = EngineConfig {
            context_parallel: 2,
            speculation: SpeculationConfig {
                enabled: true,
                ..SpeculationConfig::default()
            },
            ..EngineConfig::default()
        };
        let e = c.validate().expect_err("should be refused");
        assert!(e.contains("have not been run together"), "{e}");
    }
}

/// What this machine and this checkpoint can actually run.
///
/// A knob a user can set is not a knob a user can use. TRT-LLM's backends
/// refuse at construction time, inside the GPU allocation, with an exception
/// that names the SM version -- so the same check belongs here, where it costs
/// nothing.
///
/// This exists because it caught a live mistake. `MOE_BACKEND=DEEPGEMM` was
/// recommended in this repo on the strength of a neighbouring stack going from
/// goodput 17.0 to 19.8 with `deep_gemm`. That stack was vLLM, and vLLM's
/// DeepGEMM integration is not TRT-LLM's DeepGemmFusedMoE, which refuses
/// anything but SM100/103. The gate that was supposed to catch it checked that
/// `tensorrt_llm.deep_gemm` imports -- and it does. Import success is not
/// support.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hardware {
    /// Compute capability times 10: 90 for H200, 100 for B200.
    pub sm: u32,
    /// Checkpoint quantisation: "fp8", "nvfp4", "fp16".
    pub quant: String,
}

impl Hardware {
    /// nano4: H200 SM90, Qwen3-235B-A22B-Instruct-2507-FP8.
    pub fn h200_fp8() -> Self {
        Hardware {
            sm: 90,
            quant: "fp8".into(),
        }
    }
}

impl EngineConfig {
    /// Whether the selected kernels exist for this hardware and checkpoint.
    ///
    /// Every rule below is a `raise` in the TRT-LLM source, cited by file and
    /// line, so a reader can check the claim rather than trust this table.
    pub fn validate_for(&self, hw: &Hardware) -> Result<(), String> {
        self.validate()?;

        match self.moe_backend.as_str() {
            // fused_moe_deepgemm.py:751 -- `sm_version not in {100, 103}`.
            "DEEPGEMM" | "MEGAMOE_DEEPGEMM" if hw.sm != 100 && hw.sm != 103 => {
                return Err(format!(
                    "moe_backend = \"DEEPGEMM\" needs SM100 or SM103 and this \
                     is SM{}. TRT-LLM's DeepGemmFusedMoE refuses at \
                     construction (fused_moe_deepgemm.py:751). vLLM's \
                     deep_gemm is a different implementation and its result on \
                     this model does not transfer to this flag. On SM90 with \
                     an fp8 checkpoint, CUTLASS is the only MoE backend that \
                     runs, and AUTO already selects it.",
                    hw.sm
                ));
            }
            // fused_moe_trtllm_gen.py:157 -- same gate.
            "TRTLLM" if hw.sm != 100 && hw.sm != 103 => {
                return Err(format!(
                    "moe_backend = \"TRTLLM\" needs SM100 or SM103 and this is \
                     SM{} (fused_moe_trtllm_gen.py:157).",
                    hw.sm
                ));
            }
            // fused_moe_marlin.py:73,78 -- NVFP4 only, and SM90 only.
            "MARLIN" => {
                if hw.quant != "nvfp4" {
                    return Err(format!(
                        "moe_backend = \"MARLIN\" needs an NVFP4 checkpoint and \
                         this one is {} (fused_moe_marlin.py:73).",
                        hw.quant
                    ));
                }
                if hw.sm != 90 {
                    return Err(format!(
                        "moe_backend = \"MARLIN\" is SM90-only and this is SM{} \
                         (fused_moe_marlin.py:78).",
                        hw.sm
                    ));
                }
            }
            // fused_moe_triton.py:1499, and it additionally requires
            // swiglu_gptoss_style, which Qwen3 is not.
            "TRITON" if hw.sm != 90 => {
                return Err(format!(
                    "moe_backend = \"TRITON\" is SM90-only and this is SM{} \
                     (fused_moe_triton.py:1499).",
                    hw.sm
                ));
            }
            _ => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod hardware_tests {
    use super::*;

    /// The mistake this whole mechanism exists for, pinned so it cannot come
    /// back: DEEPGEMM was recommended for an H200 on the strength of a vLLM
    /// result, and TRT-LLM's DeepGemmFusedMoE refuses anything but SM100/103.
    #[test]
    fn deepgemm_is_refused_on_hopper_and_the_message_says_why() {
        let c = EngineConfig {
            moe_backend: "DEEPGEMM".into(),
            ..EngineConfig::default()
        };
        let e = c
            .validate_for(&Hardware::h200_fp8())
            .expect_err("DEEPGEMM must not pass on SM90");
        assert!(e.contains("SM100"), "{e}");
        assert!(
            e.contains("vLLM"),
            "the message must say why the transferred evidence does not apply: {e}"
        );
        // And it must pass where it does work, or this is just a ban.
        let b200 = Hardware {
            sm: 100,
            quant: "nvfp4".into(),
        };
        c.validate_for(&b200).expect("DEEPGEMM is fine on SM100");
    }

    /// MARLIN is SM90-only, which makes it look like the Hopper choice until
    /// you read the line above it: NVFP4 only, and this checkpoint is fp8.
    #[test]
    fn marlin_is_refused_for_the_checkpoint_not_the_gpu() {
        let c = EngineConfig {
            moe_backend: "MARLIN".into(),
            ..EngineConfig::default()
        };
        let e = c.validate_for(&Hardware::h200_fp8()).expect_err("fp8");
        assert!(
            e.contains("NVFP4"),
            "on SM90 the binding constraint is the checkpoint, not the GPU: {e}"
        );
    }

    /// What is left. AUTO resolves to CUTLASS on SM90 and CUTLASS is the only
    /// MoE backend an H200 with an fp8 checkpoint can run, so the default has
    /// no alternative to be measured against on this hardware.
    #[test]
    fn cutlass_is_the_only_moe_backend_h200_fp8_can_run() {
        let hw = Hardware::h200_fp8();
        for good in ["AUTO", "CUTLASS"] {
            EngineConfig {
                moe_backend: good.into(),
                ..EngineConfig::default()
            }
            .validate_for(&hw)
            .unwrap_or_else(|e| panic!("{good} should run on H200 fp8: {e}"));
        }
        for bad in ["DEEPGEMM", "TRTLLM", "MARLIN"] {
            assert!(
                EngineConfig {
                    moe_backend: bad.into(),
                    ..EngineConfig::default()
                }
                .validate_for(&hw)
                .is_err(),
                "{bad} must be refused on H200 fp8"
            );
        }
    }

    /// The defaults must run on the hardware this repo targets, or the first
    /// thing a user meets is an error about a file they never edited.
    #[test]
    fn the_defaults_run_on_the_target_hardware() {
        EngineConfig::default()
            .validate_for(&Hardware::h200_fp8())
            .expect("defaults must run on H200 fp8");
    }
}
