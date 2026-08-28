//! KV cache transfer.
//!
//! In a disaggregated deployment the prefill worker computes a KV cache that
//! the decode worker has to read. How that happens decides whether
//! disaggregation is a win at all, and this project has already paid to learn
//! two things about it:
//!
//! 1. **Verify that bytes moved before interpreting anything downstream.** A
//!    run where the transfer silently moved 0 bytes looks exactly like a run
//!    with a slow decode worker. Every implementation here reports
//!    [`TransferStats::bytes`], and a deployment that reads zero for it must
//!    fail loudly rather than produce a number.
//! 2. **Heterogeneous tensor parallel degrees need an explicit resharding
//!    rule.** A TP2 prefill worker feeding a TP8 decode worker does not have a
//!    one-to-one page mapping: with fewer KV heads than decode ranks, heads are
//!    *replicated* across ranks, and the destination index is an integer
//!    division, not a modulo. Getting that wrong produces plausible-looking
//!    garbage. [`Reshard`] makes the mapping a first-class value that can be
//!    unit tested without a GPU, which is the only reason to trust it.
//!
//! The 4P1D topology this repository targets is exactly the heterogeneous case:
//! four TP2 prefill workers streaming into one TP8 decode worker.

use async_trait::async_trait;
use trtllm_core::{RequestId, Result};
use trtllm_kvcache::BlockId;

pub mod local;
pub mod reshard;

pub use local::LocalTransfer;
pub use reshard::{Reshard, ReshardPlan};

/// Where a KV cache is going.
#[derive(Clone, Debug)]
pub struct TransferRequest {
    pub id: RequestId,
    pub src_blocks: Vec<BlockId>,
    pub dst_blocks: Vec<BlockId>,
    pub num_tokens: usize,
    pub reshard: Reshard,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TransferStats {
    pub bytes: u64,
    pub elapsed_ms: f64,
}

impl TransferStats {
    /// A transfer that moved nothing is a failure, whatever the return code
    /// said. Callers must check this before trusting any latency downstream.
    pub fn moved_data(&self) -> bool {
        self.bytes > 0
    }

    pub fn gib_per_s(&self) -> f64 {
        if self.elapsed_ms <= 0.0 {
            return 0.0;
        }
        (self.bytes as f64 / 1024.0 / 1024.0 / 1024.0) / (self.elapsed_ms / 1000.0)
    }
}

#[async_trait]
pub trait KvTransfer: Send + Sync {
    fn name(&self) -> &'static str;
    async fn transfer(&self, req: TransferRequest) -> Result<TransferStats>;
}
