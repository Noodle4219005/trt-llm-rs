//! Prefill queueing and batch assembly.
//!
//! The one non-obvious thing this file does is refuse to fill the batch.
//!
//! Larger prefill batches are measurably faster: at `chunked_prefill_size`
//! 16384 this model runs 4-5 sequences per forward pass and gains 11.6 % over
//! the one-sequence-per-batch behaviour that 4096 forces, because the MoE
//! grouped GEMM is more efficient at higher token counts. But every sequence in
//! a batch gets its first token at the *same* moment - the moment the whole
//! batch finishes. Batching is processor sharing, and processor sharing makes
//! every member equally late. Under a per-request deadline with a 90 % pass
//! threshold, three mediocre TTFTs beat one great one and two blown ones.
//!
//! So the batch grows while it stays *deadline feasible*: a sequence is added
//! only if the enlarged batch still finishes before the earliest first-token
//! deadline already in it. When the queue has slack the batch fills up and
//! collects the MoE efficiency; when the queue is tight it collapses towards
//! serial execution and collects the good requests instead. Nothing has to
//! switch modes - the same rule produces both.

use std::collections::HashMap;

use trtllm_core::config::PrefillPolicy;
use trtllm_core::{Millis, RequestId};

use crate::policy::{order_jobs, Job};

/// A request waiting for, or partway through, prefill.
#[derive(Clone, Debug)]
pub struct PendingPrefill {
    pub id: RequestId,
    pub arrival_ms: Millis,
    pub deadline_ms: Millis,
    /// Tokens that actually need computing, i.e. prompt length minus whatever
    /// the prefix cache supplied.
    pub compute_tokens: usize,
    /// Tokens already computed by earlier chunks.
    pub done_tokens: usize,
}

impl PendingPrefill {
    pub fn remaining(&self) -> usize {
        self.compute_tokens.saturating_sub(self.done_tokens)
    }

    pub fn is_complete(&self) -> bool {
        self.remaining() == 0
    }
}

/// One sequence's slice of a prefill batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillChunk {
    pub id: RequestId,
    /// Offset into the sequence's compute range.
    pub start: usize,
    pub tokens: usize,
    /// Whether this chunk finishes the sequence, so the first token is emitted
    /// and the request can hand off to a decode worker.
    pub completes: bool,
}

/// A batch ready to hand to the engine.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PrefillBatch {
    pub chunks: Vec<PrefillChunk>,
    pub total_tokens: usize,
    pub est_service_ms: f64,
    /// True when the batch stopped growing because of a deadline rather than
    /// because it ran out of budget or queue. Watching this flag is how you see
    /// the scheduler trading throughput for good requests.
    pub deadline_limited: bool,
}

impl PrefillBatch {
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn num_seqs(&self) -> usize {
        self.chunks.len()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PrefillTuning {
    pub chunk_tokens: usize,
    pub max_seqs: usize,
    pub policy: PrefillPolicy,
    pub demote_hopeless: bool,
    /// Aggregate prefill rate of the pool this scheduler drives, tokens/ms.
    pub tokens_per_ms: f64,
    /// Weight of a new observation in the rate EWMA.
    pub rate_alpha: f64,
}

impl Default for PrefillTuning {
    fn default() -> Self {
        Self {
            chunk_tokens: 16384,
            max_seqs: 8,
            policy: PrefillPolicy::MooreHodgson,
            demote_hopeless: true,
            // 4 x TP2 workers at ~8.5k tok/s/GPU = 68k tok/s = 68 tok/ms.
            tokens_per_ms: 68.0,
            rate_alpha: 0.2,
        }
    }
}

#[derive(Debug)]
pub struct PrefillScheduler {
    tuning: PrefillTuning,
    queue: HashMap<RequestId, PendingPrefill>,
    /// Insertion order, used only to break ties deterministically.
    order: Vec<RequestId>,
    demoted: std::collections::HashSet<RequestId>,
}

impl PrefillScheduler {
    pub fn new(tuning: PrefillTuning) -> Self {
        Self {
            tuning,
            queue: HashMap::new(),
            order: Vec::new(),
            demoted: std::collections::HashSet::new(),
        }
    }

    pub fn tuning(&self) -> &PrefillTuning {
        &self.tuning
    }

    pub fn queue_depth(&self) -> usize {
        self.queue.len()
    }

    pub fn queued_tokens(&self) -> usize {
        self.queue.values().map(PendingPrefill::remaining).sum()
    }

    /// Tokens/ms the pool is currently believed to run at.
    pub fn rate(&self) -> f64 {
        self.tuning.tokens_per_ms
    }

    pub fn enqueue(&mut self, p: PendingPrefill) {
        if self.queue.insert(p.id, p.clone()).is_none() {
            self.order.push(p.id);
        }
    }

    pub fn remove(&mut self, id: RequestId) -> Option<PendingPrefill> {
        self.order.retain(|x| *x != id);
        self.demoted.remove(&id);
        self.queue.remove(&id)
    }

    /// How many queued requests the last `plan` gave up on.
    pub fn demoted_count(&self) -> usize {
        self.demoted.len()
    }

    /// True once the request has been demoted: it cannot make its first-token
    /// deadline, so it runs behind everything that still can.
    pub fn is_demoted(&self, id: RequestId) -> bool {
        self.demoted.contains(&id)
    }

    /// Fold a completed batch back in. Sequences that finished are removed and
    /// returned so the caller can hand them to a decode worker.
    pub fn complete(&mut self, batch: &PrefillBatch, elapsed_ms: f64) -> Vec<RequestId> {
        if elapsed_ms > 0.0 && batch.total_tokens > 0 {
            let observed = batch.total_tokens as f64 / elapsed_ms;
            let a = self.tuning.rate_alpha.clamp(0.0, 1.0);
            self.tuning.tokens_per_ms = (1.0 - a) * self.tuning.tokens_per_ms + a * observed;
        }
        let mut finished = Vec::new();
        for c in &batch.chunks {
            if let Some(p) = self.queue.get_mut(&c.id) {
                p.done_tokens = (p.done_tokens + c.tokens).min(p.compute_tokens);
                if p.is_complete() {
                    finished.push(c.id);
                }
            }
        }
        for id in &finished {
            self.remove(*id);
        }
        finished
    }

    /// Assemble the next batch.
    pub fn plan(&mut self, now: Millis) -> PrefillBatch {
        if self.queue.is_empty() {
            return PrefillBatch::default();
        }
        let rate = self.tuning.tokens_per_ms.max(f64::MIN_POSITIVE);

        let jobs: Vec<Job> = self
            .order
            .iter()
            .filter_map(|id| self.queue.get(id))
            .map(|p| Job {
                id: p.id,
                arrival_ms: p.arrival_ms,
                deadline_ms: p.deadline_ms,
                service_ms: p.remaining() as f64 / rate,
            })
            .collect();

        let ordering = order_jobs(&jobs, now, self.tuning.policy);
        self.demoted = if self.tuning.demote_hopeless {
            ordering.demoted.iter().copied().collect()
        } else {
            std::collections::HashSet::new()
        };

        let mut batch = PrefillBatch::default();
        // Earliest first-token deadline among the *non-demoted* members. A
        // demoted member has already lost its deadline; letting it constrain
        // the batch would shrink the batch to one for no gain.
        let mut binding_deadline = f64::INFINITY;

        let primary: Vec<RequestId> = ordering.on_time.clone();
        let secondary: Vec<RequestId> = if self.tuning.demote_hopeless {
            ordering.demoted.clone()
        } else {
            Vec::new()
        };

        for (pass, ids) in [(false, primary), (true, secondary)] {
            for id in ids {
                if batch.chunks.len() >= self.tuning.max_seqs {
                    break;
                }
                let budget_left = self.tuning.chunk_tokens.saturating_sub(batch.total_tokens);
                if budget_left == 0 {
                    break;
                }
                let Some(p) = self.queue.get(&id) else {
                    continue;
                };
                let want = p.remaining().min(budget_left);
                if want == 0 {
                    continue;
                }

                let cand_deadline = if pass {
                    binding_deadline
                } else {
                    binding_deadline.min(p.deadline_ms)
                };
                let cand_tokens = batch.total_tokens + want;
                let finish = now + cand_tokens as f64 / rate;

                // The first sequence always goes in: refusing to run anything
                // is never better than running one thing late.
                if !batch.chunks.is_empty() && finish > cand_deadline {
                    batch.deadline_limited = true;
                    if pass {
                        break;
                    }
                    continue;
                }

                batch.chunks.push(PrefillChunk {
                    id,
                    start: p.done_tokens,
                    tokens: want,
                    completes: want == p.remaining(),
                });
                batch.total_tokens = cand_tokens;
                binding_deadline = cand_deadline;
            }
        }

        batch.est_service_ms = batch.total_tokens as f64 / rate;
        batch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tuning() -> PrefillTuning {
        PrefillTuning {
            chunk_tokens: 16384,
            max_seqs: 8,
            tokens_per_ms: 68.0,
            ..Default::default()
        }
    }

    fn pending(id: u64, arrival: f64, tokens: usize) -> PendingPrefill {
        PendingPrefill {
            id: RequestId(id),
            arrival_ms: arrival,
            deadline_ms: arrival + 3000.0,
            compute_tokens: tokens,
            done_tokens: 0,
        }
    }

    #[test]
    fn a_slack_queue_fills_the_batch_for_moe_efficiency() {
        let mut s = PrefillScheduler::new(tuning());
        for i in 0..4 {
            s.enqueue(pending(i, 0.0, 4000));
        }
        let b = s.plan(0.0);
        assert_eq!(
            b.num_seqs(),
            4,
            "16384 tokens of budget holds four 4000-token prompts"
        );
        assert_eq!(b.total_tokens, 16000);
        assert!(!b.deadline_limited);
    }

    #[test]
    fn a_tight_queue_collapses_the_batch_to_protect_deadlines() {
        let mut s = PrefillScheduler::new(tuning());
        // Everything arrived long ago; only ~150 ms of slack is left, and one
        // 4000-token prompt already costs ~59 ms.
        for i in 0..4 {
            s.enqueue(PendingPrefill {
                deadline_ms: 150.0,
                ..pending(i, -2850.0, 4000)
            });
        }
        let b = s.plan(0.0);
        assert!(
            b.num_seqs() < 4,
            "batch must shrink under deadline pressure: {b:?}"
        );
        assert!(b.deadline_limited);
        assert!(b.est_service_ms <= 150.0);
    }

    #[test]
    fn the_first_sequence_always_runs_even_when_hopeless() {
        let mut s = PrefillScheduler::new(tuning());
        s.enqueue(PendingPrefill {
            deadline_ms: -1.0,
            ..pending(0, -5000.0, 4000)
        });
        let b = s.plan(0.0);
        assert_eq!(b.num_seqs(), 1);
        assert_eq!(b.chunks[0].id, RequestId(0));
    }

    #[test]
    fn long_prompts_are_chunked_and_only_the_last_chunk_completes() {
        let mut s = PrefillScheduler::new(PrefillTuning {
            chunk_tokens: 4096,
            ..tuning()
        });
        s.enqueue(pending(0, 0.0, 10_000));
        let b1 = s.plan(0.0);
        assert_eq!(b1.total_tokens, 4096);
        assert!(!b1.chunks[0].completes);
        s.complete(&b1, b1.est_service_ms);

        let b2 = s.plan(100.0);
        assert_eq!(b2.chunks[0].start, 4096);
        assert!(!b2.chunks[0].completes);
        s.complete(&b2, b2.est_service_ms);

        let b3 = s.plan(200.0);
        assert_eq!(b3.total_tokens, 10_000 - 2 * 4096);
        assert!(b3.chunks[0].completes);
        let done = s.complete(&b3, b3.est_service_ms);
        assert_eq!(done, vec![RequestId(0)]);
        assert_eq!(s.queue_depth(), 0);
    }

    #[test]
    fn observed_throughput_updates_the_rate_estimate() {
        let mut s = PrefillScheduler::new(tuning());
        s.enqueue(pending(0, 0.0, 4000));
        let b = s.plan(0.0);
        let before = s.rate();
        // The pool actually ran at half the assumed rate.
        s.complete(&b, b.est_service_ms * 2.0);
        assert!(
            s.rate() < before,
            "rate estimate must fall: {} -> {}",
            before,
            s.rate()
        );
    }

    #[test]
    fn a_hopeless_request_does_not_block_the_ones_that_can_still_be_good() {
        let mut s = PrefillScheduler::new(tuning());
        // id 0 is already past its deadline; ids 1..3 have plenty of slack.
        s.enqueue(PendingPrefill {
            deadline_ms: -100.0,
            ..pending(0, -4000.0, 4000)
        });
        for i in 1..4 {
            s.enqueue(pending(i, 0.0, 4000));
        }
        let b = s.plan(0.0);
        let head = b.chunks[0].id;
        assert_ne!(
            head,
            RequestId(0),
            "the doomed request must not hold the head of the batch"
        );
        assert!(s.is_demoted(RequestId(0)));
    }
}
