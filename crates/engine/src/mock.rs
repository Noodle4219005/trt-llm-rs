//! An engine that costs work instead of running it.
//!
//! This is not a toy. It is the reason a scheduling change can be rejected for
//! free: the mock runs the *real* control plane - the same schedulers, the same
//! router, the same admission rules - against a cost model fitted to measured
//! hardware. Only the kernels are missing.
//!
//! Two time modes:
//!
//! * [`TimeMode::Virtual`] returns the computed duration without waiting. Used
//!   by the simulator and by tests; a 120-second benchmark window runs in
//!   milliseconds and is bit-for-bit reproducible.
//! * [`TimeMode::Wall`] actually sleeps. Used to exercise the HTTP frontend and
//!   the router end to end on a machine with no GPUs.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use tokio::sync::Mutex;
use trtllm_core::{Millis, Phase, RequestId, Result, TokenId};

use crate::cost::CostModel;
use crate::{DecodeSeqSpec, DecodeStepOutcome, Engine, EngineInfo, PrefillOutcome, PrefillWork};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeMode {
    Virtual,
    Wall,
}

#[derive(Debug)]
struct MockSeq {
    spec: DecodeSeqSpec,
    emitted: u32,
}

#[derive(Debug)]
pub struct MockEngine {
    info: EngineInfo,
    cost: CostModel,
    mode: TimeMode,
    decode: Mutex<HashMap<RequestId, MockSeq>>,
    steps: AtomicU64,
}

impl MockEngine {
    pub fn new(info: EngineInfo, cost: CostModel, mode: TimeMode) -> Self {
        Self {
            info,
            cost,
            mode,
            decode: Mutex::new(HashMap::new()),
            steps: AtomicU64::new(0),
        }
    }

    pub fn cost(&self) -> &CostModel {
        &self.cost
    }

    pub fn steps(&self) -> u64 {
        self.steps.load(Ordering::Relaxed)
    }

    async fn spend(&self, ms: f64) {
        if self.mode == TimeMode::Wall && ms > 0.0 {
            tokio::time::sleep(std::time::Duration::from_secs_f64(ms / 1000.0)).await;
        }
    }
}

/// Deterministic stand-in for a sampled token: never a real distribution, but
/// stable across runs so a diff in the output means a diff in the scheduling.
fn fake_token(id: RequestId, position: u32) -> TokenId {
    let h =
        id.0.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(u64::from(position));
    ((h >> 32) as u32) % 150_000
}

#[async_trait]
impl Engine for MockEngine {
    fn info(&self) -> EngineInfo {
        self.info.clone()
    }

    async fn prefill(&self, work: PrefillWork, _now: Millis) -> Result<PrefillOutcome> {
        let tokens = work.total_tokens();
        let ms = self.cost.prefill.batch_ms(tokens, work.chunks.len());
        self.spend(ms).await;
        let first_tokens = work
            .chunks
            .iter()
            .filter(|c| c.completes)
            .map(|c| (c.id, fake_token(c.id, 0)))
            .collect();
        Ok(PrefillOutcome {
            elapsed_ms: ms,
            first_tokens,
        })
    }

    async fn add_decode_seq(&self, seq: DecodeSeqSpec) -> Result<()> {
        self.decode.lock().await.insert(
            seq.id,
            MockSeq {
                spec: seq,
                emitted: 1,
            },
        );
        Ok(())
    }

    async fn decode_step(&self, _now: Millis) -> Result<DecodeStepOutcome> {
        let mut guard = self.decode.lock().await;
        let concurrency = guard.len();
        if concurrency == 0 {
            return Ok(DecodeStepOutcome::default());
        }
        let ms = self.cost.decode.step_ms(concurrency);

        let mut out = DecodeStepOutcome {
            elapsed_ms: ms,
            ..Default::default()
        };
        for seq in guard.values_mut() {
            seq.emitted += 1;
            out.tokens
                .push((seq.spec.id, fake_token(seq.spec.id, seq.emitted)));
            if seq.emitted >= seq.spec.max_tokens {
                out.finished.push(seq.spec.id);
            }
        }
        for id in &out.finished {
            guard.remove(id);
        }
        drop(guard);

        self.steps.fetch_add(1, Ordering::Relaxed);
        self.spend(ms).await;
        Ok(out)
    }

    async fn remove_seq(&self, id: RequestId) -> Result<()> {
        self.decode.lock().await.remove(&id);
        Ok(())
    }

    async fn decode_concurrency(&self) -> usize {
        self.decode.lock().await.len()
    }
}

/// Convenience constructor for a prefill worker.
pub fn mock_prefill_worker(
    model: &str,
    gpus: Vec<u32>,
    tp: u32,
    cost: CostModel,
    mode: TimeMode,
) -> MockEngine {
    MockEngine::new(
        EngineInfo {
            backend: "mock".into(),
            model: model.into(),
            phase: Phase::Prefill,
            tensor_parallel: tp,
            gpus,
            kv_blocks: 4096,
            kv_block_size: 128,
        },
        cost,
        mode,
    )
}

/// Convenience constructor for a decode worker.
pub fn mock_decode_worker(
    model: &str,
    gpus: Vec<u32>,
    tp: u32,
    kv_blocks: u32,
    cost: CostModel,
    mode: TimeMode,
) -> MockEngine {
    MockEngine::new(
        EngineInfo {
            backend: "mock".into(),
            model: model.into(),
            phase: Phase::Decode,
            tensor_parallel: tp,
            gpus,
            kv_blocks,
            kv_block_size: 128,
        },
        cost,
        mode,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::{DecodeCurve, PrefillCurve};
    use crate::PrefillChunkSpec;
    use trtllm_core::capacity::{DecodeCalibration, PrefillCalibration};

    fn cost() -> CostModel {
        CostModel::new(
            PrefillCurve::for_worker(&PrefillCalibration::default(), 2, 2),
            DecodeCurve::from_calibration(&DecodeCalibration::default(), 8),
            0.0,
        )
    }

    fn chunk(id: u64, tokens: usize, completes: bool) -> PrefillChunkSpec {
        PrefillChunkSpec {
            id: RequestId(id),
            tokens: vec![7; tokens],
            start: 0,
            prompt_len: tokens,
            completes,
            kv_blocks: Vec::new(),
        }
    }

    #[tokio::test]
    async fn only_completed_chunks_emit_a_first_token() {
        let e = mock_prefill_worker("m", vec![0, 1], 2, cost(), TimeMode::Virtual);
        let work = PrefillWork {
            chunks: vec![chunk(1, 4000, true), chunk(2, 4000, false)],
        };
        let out = e.prefill(work, 0.0).await.expect("prefill");
        assert_eq!(out.first_tokens.len(), 1);
        assert_eq!(out.first_tokens[0].0, RequestId(1));
        assert!(out.elapsed_ms > 0.0);
    }

    #[tokio::test]
    async fn decode_steps_cost_more_as_the_batch_grows() {
        let e = mock_decode_worker("m", (0..8).collect(), 8, 8192, cost(), TimeMode::Virtual);
        e.add_decode_seq(DecodeSeqSpec {
            id: RequestId(1),
            prompt_len: 4000,
            first_token: 1,
            max_tokens: 200,
            kv_blocks: Vec::new(),
        })
        .await
        .expect("add");
        let one = e.decode_step(0.0).await.expect("step").elapsed_ms;

        for i in 2..40u64 {
            e.add_decode_seq(DecodeSeqSpec {
                id: RequestId(i),
                prompt_len: 4000,
                first_token: 1,
                max_tokens: 200,
                kv_blocks: Vec::new(),
            })
            .await
            .expect("add");
        }
        let many = e.decode_step(0.0).await.expect("step").elapsed_ms;
        assert!(many > one, "{many} should exceed {one}");
        assert_eq!(e.decode_concurrency().await, 39);
    }

    #[tokio::test]
    async fn a_sequence_retires_after_its_token_budget() {
        let e = mock_decode_worker("m", (0..8).collect(), 8, 8192, cost(), TimeMode::Virtual);
        e.add_decode_seq(DecodeSeqSpec {
            id: RequestId(1),
            prompt_len: 4000,
            first_token: 1,
            max_tokens: 3,
            kv_blocks: Vec::new(),
        })
        .await
        .expect("add");
        assert!(e.decode_step(0.0).await.expect("step").finished.is_empty());
        let out = e.decode_step(0.0).await.expect("step");
        assert_eq!(out.finished, vec![RequestId(1)]);
        assert_eq!(e.decode_concurrency().await, 0);
    }

    #[tokio::test]
    async fn an_idle_decode_engine_costs_nothing() {
        let e = mock_decode_worker("m", (0..8).collect(), 8, 8192, cost(), TimeMode::Virtual);
        let out = e.decode_step(0.0).await.expect("step");
        assert_eq!(out.elapsed_ms, 0.0);
        assert!(out.tokens.is_empty());
    }
}
