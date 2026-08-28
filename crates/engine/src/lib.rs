//! Execution backends.
//!
//! The control plane never talks to CUDA. It talks to [`Engine`], and there are
//! two implementations:
//!
//! * [`mock::MockEngine`] costs a batch from the calibrated model. It runs on a
//!   laptop, it is deterministic, and it is what lets a scheduling policy be
//!   accepted or rejected before anyone spends a GPU-hour on it.
//! * [`trtllm::TrtllmEngine`] binds the TensorRT-LLM C++ Executor API. It is
//!   behind the `trtllm` feature, needs a CUDA toolchain, and is **not built or
//!   tested in this tree** - see `docs/trtllm-ffi.md` for the shim it expects.
//!
//! Keeping the seam this narrow is the whole point of the rewrite. Everything
//! that decides *which* tokens run and *when* is Rust; everything that decides
//! how fast a GEMM goes stays in the kernels that already do it well.

pub mod cost;
pub mod mock;
#[cfg(feature = "trtllm")]
pub mod trtllm;

use async_trait::async_trait;
use trtllm_core::{Millis, Phase, RequestId};
use trtllm_core::{Result, TokenId};
use trtllm_kvcache::BlockId;

pub use cost::CostModel;
pub use mock::MockEngine;

/// Static description of a running engine.
#[derive(Clone, Debug)]
pub struct EngineInfo {
    pub backend: String,
    pub model: String,
    pub phase: Phase,
    pub tensor_parallel: u32,
    pub gpus: Vec<u32>,
    /// Blocks in this worker's KV pool.
    pub kv_blocks: u32,
    pub kv_block_size: u32,
}

/// One sequence's slice of a prefill batch, resolved down to real tokens and
/// the pages its KV will land in.
#[derive(Clone, Debug)]
pub struct PrefillChunkSpec {
    pub id: RequestId,
    /// The tokens this chunk computes, already offset by `start`.
    pub tokens: Vec<TokenId>,
    pub start: usize,
    /// Total prompt length, needed to size the KV allocation up front.
    pub prompt_len: usize,
    pub completes: bool,
    pub kv_blocks: Vec<BlockId>,
}

#[derive(Clone, Debug, Default)]
pub struct PrefillWork {
    pub chunks: Vec<PrefillChunkSpec>,
}

impl PrefillWork {
    pub fn total_tokens(&self) -> usize {
        self.chunks.iter().map(|c| c.tokens.len()).sum()
    }
}

#[derive(Clone, Debug, Default)]
pub struct PrefillOutcome {
    pub elapsed_ms: f64,
    /// First sampled token for each sequence whose prefill completed in this
    /// batch. Sequences still being chunked do not appear.
    pub first_tokens: Vec<(RequestId, TokenId)>,
}

/// A sequence being handed to a decode engine, with the KV already in place.
#[derive(Clone, Debug)]
pub struct DecodeSeqSpec {
    pub id: RequestId,
    pub prompt_len: usize,
    pub first_token: TokenId,
    pub max_tokens: u32,
    pub kv_blocks: Vec<BlockId>,
}

#[derive(Clone, Debug, Default)]
pub struct DecodeStepOutcome {
    pub elapsed_ms: f64,
    pub tokens: Vec<(RequestId, TokenId)>,
    pub finished: Vec<RequestId>,
}

/// The seam between the Rust control plane and whatever actually runs the model.
#[async_trait]
pub trait Engine: Send + Sync {
    fn info(&self) -> EngineInfo;

    /// Execute one prefill batch. Implementations must return the *measured*
    /// elapsed time: the prefill scheduler feeds it straight into its rate
    /// estimate, and a fabricated number there quietly corrupts every
    /// subsequent deadline decision.
    async fn prefill(&self, work: PrefillWork, now: Millis) -> Result<PrefillOutcome>;

    /// Attach a sequence whose prefill is already done.
    async fn add_decode_seq(&self, seq: DecodeSeqSpec) -> Result<()>;

    /// Run one decode forward pass over every attached sequence.
    async fn decode_step(&self, now: Millis) -> Result<DecodeStepOutcome>;

    /// Detach a sequence, on completion or cancellation.
    async fn remove_seq(&self, id: RequestId) -> Result<()>;

    /// Sequences currently attached to the decode batch.
    async fn decode_concurrency(&self) -> usize;
}
