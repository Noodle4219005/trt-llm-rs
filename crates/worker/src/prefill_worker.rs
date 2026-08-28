use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::mpsc;
use trtllm_core::{Millis, RequestId, TokenId, WorkerId};
use trtllm_engine::{Engine, PrefillChunkSpec, PrefillWork};
use trtllm_kvcache::{blocks_for, BlockPool};
use trtllm_sched::prefill::PrefillTuning;
use trtllm_sched::{PendingPrefill, PrefillScheduler};

/// A request handed to a prefill worker.
pub struct PrefillJob {
    pub id: RequestId,
    pub prompt: Vec<TokenId>,
    pub deadline_ms: Millis,
    pub arrival_ms: Millis,
}

/// What the worker emits when a prefill finishes: the first token, sampled on
/// the prefill worker itself so time-to-first-token stops here rather than
/// after the KV handoff.
#[derive(Clone, Debug)]
pub struct PrefillComplete {
    pub id: RequestId,
    pub first_token: TokenId,
    pub prompt_len: usize,
    pub finished_at_ms: Millis,
}

pub struct PrefillWorker {
    pub id: WorkerId,
    engine: Arc<dyn Engine>,
    sched: Mutex<PrefillScheduler>,
    pool: Mutex<BlockPool>,
    block_size: u32,
    prompts: Mutex<std::collections::HashMap<RequestId, Vec<TokenId>>>,
    /// Pages already holding computed KV for a request. Chunked prefill spans
    /// several forward passes, and the earlier chunks' KV has to survive until
    /// the last one - freeing per chunk would recompute the prompt silently.
    kv: Mutex<std::collections::HashMap<RequestId, Vec<trtllm_kvcache::BlockId>>>,
    epoch: std::time::Instant,
}

impl PrefillWorker {
    pub fn new(
        id: WorkerId,
        engine: Arc<dyn Engine>,
        tuning: PrefillTuning,
        num_blocks: u32,
        block_size: u32,
        watermark: f64,
        epoch: std::time::Instant,
    ) -> Self {
        Self {
            id,
            engine,
            sched: Mutex::new(PrefillScheduler::new(tuning)),
            pool: Mutex::new(BlockPool::new(num_blocks, block_size, watermark)),
            block_size,
            prompts: Mutex::new(std::collections::HashMap::new()),
            kv: Mutex::new(std::collections::HashMap::new()),
            epoch,
        }
    }

    pub fn now_ms(&self) -> Millis {
        self.epoch.elapsed().as_secs_f64() * 1000.0
    }

    pub fn queued_tokens(&self) -> usize {
        self.sched.lock().queued_tokens()
    }

    pub fn queue_depth(&self) -> usize {
        self.sched.lock().queue_depth()
    }

    pub fn rate(&self) -> f64 {
        self.sched.lock().rate()
    }

    fn enqueue(&self, job: PrefillJob) {
        let len = job.prompt.len();
        self.prompts.lock().insert(job.id, job.prompt);
        self.sched.lock().enqueue(PendingPrefill {
            id: job.id,
            arrival_ms: job.arrival_ms,
            deadline_ms: job.deadline_ms,
            compute_tokens: len,
            done_tokens: 0,
        });
    }

    /// Run until the job channel closes.
    pub async fn run(
        self: Arc<Self>,
        mut jobs: mpsc::Receiver<PrefillJob>,
        out: mpsc::Sender<PrefillComplete>,
    ) {
        loop {
            if self.queue_depth() == 0 {
                match jobs.recv().await {
                    Some(job) => self.enqueue(job),
                    None => break,
                }
            }
            while let Ok(job) = jobs.try_recv() {
                self.enqueue(job);
            }

            let now = self.now_ms();
            let batch = self.sched.lock().plan(now);
            if batch.is_empty() {
                continue;
            }

            // Resolve the batch into real tokens and pages. A chunk that cannot
            // get KV is dropped from this batch and retried next round rather
            // than failing the request: the scheduler already owns the queue.
            let mut work = PrefillWork::default();
            {
                let prompts = self.prompts.lock();
                let mut kv = self.kv.lock();
                let mut pool = self.pool.lock();
                for c in &batch.chunks {
                    let Some(prompt) = prompts.get(&c.id) else {
                        continue;
                    };
                    let end = (c.start + c.tokens).min(prompt.len());
                    let held = kv.get(&c.id).map_or(0, Vec::len);
                    let want = blocks_for(end, self.block_size);
                    let extra = want.saturating_sub(held);
                    if extra > 0 {
                        if !pool.can_admit(extra) {
                            break;
                        }
                        let Ok(blocks) = pool.alloc(extra) else { break };
                        kv.entry(c.id).or_default().extend(blocks);
                    }
                    work.chunks.push(PrefillChunkSpec {
                        id: c.id,
                        tokens: prompt[c.start..end].to_vec(),
                        start: c.start,
                        prompt_len: prompt.len(),
                        completes: c.completes,
                        kv_blocks: kv.get(&c.id).cloned().unwrap_or_default(),
                    });
                }
            }
            if work.chunks.is_empty() {
                tokio::task::yield_now().await;
                continue;
            }

            let outcome = match self.engine.prefill(work, now).await {
                Ok(o) => o,
                Err(e) => {
                    tracing::error!(worker = %self.id, error = %e, "prefill batch failed");
                    continue;
                }
            };

            let finished = self.sched.lock().complete(&batch, outcome.elapsed_ms);
            let done_at = self.now_ms();

            for (id, token) in outcome.first_tokens {
                let prompt_len = self
                    .prompts
                    .lock()
                    .get(&id)
                    .map(Vec::len)
                    .unwrap_or_default();
                if out
                    .send(PrefillComplete {
                        id,
                        first_token: token,
                        prompt_len,
                        finished_at_ms: done_at,
                    })
                    .await
                    .is_err()
                {
                    return;
                }
            }

            // Once a sequence is complete its KV belongs to the transfer, not
            // to this pool. A real NIXL path holds the pages until the receiver
            // acknowledges; releasing here models a copy that has landed.
            {
                let mut kv = self.kv.lock();
                let mut pool = self.pool.lock();
                for id in &finished {
                    if let Some(blocks) = kv.remove(id) {
                        pool.release(&blocks);
                    }
                }
            }
            for id in &finished {
                self.prompts.lock().remove(id);
            }
        }
    }
}
