//! In-process transfer, used by the simulator and by GPU-free end-to-end runs.

use async_trait::async_trait;
use trtllm_core::Result;

use crate::{KvTransfer, TransferRequest, TransferStats};

/// Copies nothing but reports exactly what a real transfer would have moved,
/// so the "did any bytes move?" check has something honest to read.
#[derive(Clone, Copy, Debug)]
pub struct LocalTransfer {
    /// Bytes of KV per token, both K and V, across all layers and heads.
    pub bytes_per_token: u64,
    /// Modelled fabric bandwidth, GiB/s. NVLink within a node, IB across.
    pub bandwidth_gib_s: f64,
    /// Fixed handshake cost.
    pub latency_ms: f64,
}

impl LocalTransfer {
    /// KV bytes per token for one model configuration.
    /// `2 (K and V) * layers * kv_heads * head_dim * dtype_bytes`.
    pub fn bytes_per_token(layers: u32, kv_heads: u32, head_dim: u32, dtype_bytes: u32) -> u64 {
        2 * u64::from(layers) * u64::from(kv_heads) * u64::from(head_dim) * u64::from(dtype_bytes)
    }
}

impl Default for LocalTransfer {
    /// Qwen3-235B-A22B: 94 layers, 4 KV heads, head dim 128, FP8 KV.
    /// That is 96 KiB per token, so a 4000-token prompt moves ~376 MiB.
    fn default() -> Self {
        Self {
            bytes_per_token: Self::bytes_per_token(94, 4, 128, 1),
            bandwidth_gib_s: 40.0,
            latency_ms: 0.5,
        }
    }
}

#[async_trait]
impl KvTransfer for LocalTransfer {
    fn name(&self) -> &'static str {
        "local"
    }

    async fn transfer(&self, req: TransferRequest) -> Result<TransferStats> {
        let plan = req.reshard.plan()?;
        let bytes = (req.num_tokens as u64) * self.bytes_per_token;
        let on_wire = bytes as f64 * plan.amplification();
        let secs = on_wire / (self.bandwidth_gib_s * 1024.0 * 1024.0 * 1024.0);
        Ok(TransferStats {
            bytes,
            elapsed_ms: self.latency_ms + secs * 1000.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Reshard;
    use trtllm_core::RequestId;

    #[tokio::test]
    async fn a_4000_token_prompt_moves_about_366_mib() {
        let t = LocalTransfer::default();
        let stats = t
            .transfer(TransferRequest {
                id: RequestId(1),
                src_blocks: Vec::new(),
                dst_blocks: Vec::new(),
                num_tokens: 4000,
                reshard: Reshard::identity(4, 8),
            })
            .await
            .expect("transfer");
        assert!(stats.moved_data());
        let mib = stats.bytes as f64 / 1024.0 / 1024.0;
        assert!((mib - 366.0).abs() < 2.0, "{mib} MiB");
        assert!(stats.elapsed_ms > 0.0);
    }

    #[tokio::test]
    async fn replication_costs_wire_time_but_not_payload_bytes() {
        let t = LocalTransfer::default();
        let mk = |reshard| TransferRequest {
            id: RequestId(1),
            src_blocks: Vec::new(),
            dst_blocks: Vec::new(),
            num_tokens: 4000,
            reshard,
        };
        let same = t.transfer(mk(Reshard::identity(4, 4))).await.expect("same");
        let hetero = t
            .transfer(mk(Reshard {
                num_kv_heads: 4,
                src_tp: 2,
                dst_tp: 8,
            }))
            .await
            .expect("het");
        assert_eq!(same.bytes, hetero.bytes);
        assert!(hetero.elapsed_ms > same.elapsed_ms);
    }
}
