use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::mpsc;
use trtllm_core::{Millis, RequestId, TokenId, WorkerId};
use trtllm_engine::{DecodeSeqSpec, Engine};
use trtllm_kvcache::{blocks_for, BlockId, BlockPool};
use trtllm_sched::{DecodeScheduler, RunningSeq};

/// A sequence whose prefill is done and whose KV has landed.
#[derive(Clone, Debug)]
pub struct DecodeAdmit {
    pub id: RequestId,
    pub first_token: TokenId,
    pub prompt_len: usize,
    pub max_tokens: u32,
    pub first_token_ms: Millis,
}

#[derive(Clone, Debug)]
pub enum DecodeEvent {
    Token { id: RequestId, token: TokenId },
    Finished { id: RequestId, tokens_emitted: u32 },
}

pub struct DecodeWorker {
    pub id: WorkerId,
    engine: Arc<dyn Engine>,
    sched: Mutex<DecodeScheduler>,
    pool: Mutex<BlockPool>,
    kv: Mutex<HashMap<RequestId, Vec<BlockId>>>,
    block_size: u32,
    epoch: std::time::Instant,
}

impl DecodeWorker {
    pub fn new(
        id: WorkerId,
        engine: Arc<dyn Engine>,
        sched: DecodeScheduler,
        num_blocks: u32,
        block_size: u32,
        watermark: f64,
        epoch: std::time::Instant,
    ) -> Self {
        Self {
            id,
            engine,
            sched: Mutex::new(sched),
            pool: Mutex::new(BlockPool::new(num_blocks, block_size, watermark)),
            kv: Mutex::new(HashMap::new()),
            block_size,
            epoch,
        }
    }

    pub fn now_ms(&self) -> Millis {
        self.epoch.elapsed().as_secs_f64() * 1000.0
    }

    pub fn concurrency(&self) -> usize {
        self.sched.lock().concurrency()
    }

    pub fn cap(&self) -> f64 {
        self.sched.lock().controller().cap()
    }

    pub fn observed_step_ms(&self) -> f64 {
        self.sched.lock().controller().observed_itl_ms()
    }

    pub fn kv_utilisation(&self) -> f64 {
        self.pool.lock().stats().utilisation()
    }

    pub fn refusals(&self) -> u64 {
        self.sched.lock().refused()
    }

    pub async fn run(
        self: Arc<Self>,
        mut jobs: mpsc::Receiver<DecodeAdmit>,
        out: mpsc::Sender<DecodeEvent>,
    ) {
        let mut pending: VecDeque<DecodeAdmit> = VecDeque::new();
        loop {
            while let Ok(j) = jobs.try_recv() {
                pending.push_back(j);
            }
            if self.concurrency() == 0 && pending.is_empty() {
                match jobs.recv().await {
                    Some(j) => pending.push_back(j),
                    None => break,
                }
            }

            // Admission. Every refusal is recorded with its reason so a stalled
            // decode pool can be attributed to the cap, to an in-flight
            // request's ITL average, or to KV - never to "it was slow".
            while let Some(job) = pending.front().cloned() {
                let need = blocks_for(job.prompt_len + job.max_tokens as usize, self.block_size);
                let headroom = self.pool.lock().can_admit(need);
                let decision = {
                    let mut s = self.sched.lock();
                    let d = s.can_admit(headroom);
                    s.note(d);
                    d
                };
                if !decision.is_admit() {
                    break;
                }
                let Ok(blocks) = self.pool.lock().alloc(need) else {
                    break;
                };
                self.kv.lock().insert(job.id, blocks.clone());
                if let Err(e) = self
                    .engine
                    .add_decode_seq(DecodeSeqSpec {
                        id: job.id,
                        prompt_len: job.prompt_len,
                        first_token: job.first_token,
                        max_tokens: job.max_tokens,
                        kv_blocks: blocks.clone(),
                    })
                    .await
                {
                    tracing::error!(worker = %self.id, error = %e, "add_decode_seq failed");
                    self.pool.lock().release(&blocks);
                    self.kv.lock().remove(&job.id);
                    pending.pop_front();
                    continue;
                }
                self.sched.lock().admit(RunningSeq::new(
                    job.id,
                    job.first_token_ms,
                    job.max_tokens,
                ));
                pending.pop_front();
            }

            if self.concurrency() == 0 {
                tokio::task::yield_now().await;
                continue;
            }

            let now = self.now_ms();
            let step = match self.engine.decode_step(now).await {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!(worker = %self.id, error = %e, "decode step failed");
                    continue;
                }
            };
            let done = self.sched.lock().on_step(self.now_ms(), step.elapsed_ms);

            for (id, token) in step.tokens {
                if out.send(DecodeEvent::Token { id, token }).await.is_err() {
                    return;
                }
            }
            for seq in done {
                if let Some(blocks) = self.kv.lock().remove(&seq.id) {
                    self.pool.lock().release(&blocks);
                }
                let _ = self.engine.remove_seq(seq.id).await;
                if out
                    .send(DecodeEvent::Finished {
                        id: seq.id,
                        tokens_emitted: seq.tokens_emitted,
                    })
                    .await
                    .is_err()
                {
                    return;
                }
            }
        }
    }
}
