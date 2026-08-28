use trtllm_core::{Millis, TokenId, WorkerId};
use trtllm_kvcache::RadixIndex;

use crate::registry::{WorkerRegistry, WorkerRole};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutingError {
    NoPrefillWorker,
    NoDecodeWorker,
}

impl RoutingError {
    pub fn as_str(self) -> &'static str {
        match self {
            RoutingError::NoPrefillWorker => "no healthy prefill worker",
            RoutingError::NoDecodeWorker => "no healthy decode worker",
        }
    }
}

/// The chosen pair plus the arithmetic that chose them, so a bad routing
/// decision can be read off a log rather than reproduced.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoutingDecision {
    pub prefill: WorkerId,
    pub decode: WorkerId,
    /// Tokens the prefill worker will not have to compute thanks to reuse.
    pub prefix_hit_tokens: usize,
    /// Predicted milliseconds until the first token, at the moment of routing.
    pub predicted_ttft_ms: f64,
    /// Whether that prediction already exceeds the request's budget. The
    /// scheduler uses this to decide the request is a lost cause before it
    /// occupies the head of a prefill batch.
    pub predicted_late: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct RouterTuning {
    /// Residual KV handoff cost not overlapped with prefill, milliseconds.
    pub kv_transfer_ms: f64,
    /// Rate assumed for a worker that has not reported one yet, tokens/ms.
    pub default_tokens_per_ms: f64,
    /// Minimum prefix length worth routing for.
    pub min_prefix_tokens: usize,
}

impl Default for RouterTuning {
    fn default() -> Self {
        Self {
            kv_transfer_ms: 10.0,
            default_tokens_per_ms: 17.0,
            min_prefix_tokens: 256,
        }
    }
}

#[derive(Debug)]
pub struct Router {
    pub registry: WorkerRegistry,
    pub prefix_index: RadixIndex,
    tuning: RouterTuning,
}

impl Router {
    pub fn new(registry: WorkerRegistry, tuning: RouterTuning) -> Self {
        Self {
            registry,
            prefix_index: RadixIndex::new(tuning.min_prefix_tokens, 65_536),
            tuning,
        }
    }

    pub fn tuning(&self) -> &RouterTuning {
        &self.tuning
    }

    /// Pick a prefill and a decode worker for one prompt.
    pub fn route(
        &self,
        prompt: &[TokenId],
        now: Millis,
        ttft_budget_ms: f64,
    ) -> Result<RoutingDecision, RoutingError> {
        let prefill_pool = self.registry.live(WorkerRole::Prefill, now);
        if prefill_pool.is_empty() {
            return Err(RoutingError::NoPrefillWorker);
        }
        let decode_pool = self.registry.live(WorkerRole::Decode, now);
        if decode_pool.is_empty() {
            return Err(RoutingError::NoDecodeWorker);
        }

        let overlaps = self.prefix_index.match_workers(prompt);

        let mut best: Option<(f64, WorkerId, usize)> = None;
        for w in &prefill_pool {
            let rate = if w.load.tokens_per_ms > 0.0 {
                w.load.tokens_per_ms
            } else {
                self.tuning.default_tokens_per_ms
            };
            let hit = overlaps.get(&w.id).copied().unwrap_or(0).min(prompt.len());
            let compute = prompt.len().saturating_sub(hit) as f64;
            let wait = w.load.queued_tokens as f64 / rate;
            let cost = wait + compute / rate + self.tuning.kv_transfer_ms;
            // Ties go to the lower worker id so a cold fleet does not herd
            // onto whichever worker happens to hash first.
            let better = match &best {
                None => true,
                Some((c, id, _)) => cost < *c - 1e-9 || ((cost - *c).abs() <= 1e-9 && w.id < *id),
            };
            if better {
                best = Some((cost, w.id, hit));
            }
        }
        let (predicted_ttft_ms, prefill, prefix_hit_tokens) = best.expect("pool is non-empty");

        // Decode: the worker with the most room under its own admission cap,
        // breaking ties on KV utilisation. Slots run out before KV does on this
        // workload, so slots are the primary key.
        let decode = decode_pool
            .iter()
            .max_by(|a, b| {
                a.decode_headroom()
                    .total_cmp(&b.decode_headroom())
                    .then(b.load.kv_utilisation.total_cmp(&a.load.kv_utilisation))
                    .then(b.id.cmp(&a.id))
            })
            .map(|w| w.id)
            .expect("pool is non-empty");

        Ok(RoutingDecision {
            prefill,
            decode,
            prefix_hit_tokens,
            predicted_ttft_ms,
            predicted_late: predicted_ttft_ms > ttft_budget_ms,
        })
    }

    /// Tell the index that a worker now holds this prompt's KV.
    pub fn note_prefill(&mut self, worker: WorkerId, prompt: &[TokenId], now: Millis) {
        self.prefix_index.insert(worker, prompt, now);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{WorkerLoad, WorkerState};

    fn worker(id: u32, role: WorkerRole, queued: u64, rate: f64) -> WorkerState {
        WorkerState {
            id: WorkerId(id),
            role,
            endpoint: format!("http://w{id}"),
            tensor_parallel: 2,
            healthy: true,
            load: WorkerLoad {
                queued_tokens: queued,
                tokens_per_ms: rate,
                decode_cap: 64.0,
                updated_ms: 0.0,
                ..Default::default()
            },
        }
    }

    fn registry(prefill: &[(u32, u64, f64)]) -> WorkerRegistry {
        let mut r = WorkerRegistry::new(5_000.0);
        for (id, queued, rate) in prefill {
            r.register(worker(*id, WorkerRole::Prefill, *queued, *rate));
        }
        r.register(worker(100, WorkerRole::Decode, 0, 0.0));
        r
    }

    #[test]
    fn the_shortest_predicted_wait_wins() {
        let router = Router::new(
            registry(&[(1, 40_000, 17.0), (2, 0, 17.0)]),
            RouterTuning::default(),
        );
        let d = router.route(&vec![1u32; 4000], 0.0, 3000.0).expect("route");
        assert_eq!(d.prefill, WorkerId(2));
        assert_eq!(d.decode, WorkerId(100));
        assert!(!d.predicted_late);
    }

    /// A prefix hit is worth exactly the milliseconds of prefill it saves, so
    /// it can outweigh a queue - but only when it actually is longer than the
    /// queue it has to jump.
    #[test]
    fn a_prefix_hit_is_priced_in_milliseconds_not_bonus_points() {
        let mut router = Router::new(
            registry(&[(1, 20_000, 17.0), (2, 0, 17.0)]),
            RouterTuning::default(),
        );
        let prompt = vec![5u32; 4000];
        // Worker 1 holds the whole prompt: it saves 4000 tokens of compute but
        // carries 20000 tokens of queue, so it should still lose.
        router.note_prefill(WorkerId(1), &prompt, 0.0);
        assert_eq!(
            router.route(&prompt, 0.0, 3000.0).expect("route").prefill,
            WorkerId(2)
        );

        // Shrink the queue below what the hit saves and it wins.
        router.registry.update_load(
            WorkerId(1),
            WorkerLoad {
                queued_tokens: 1_000,
                tokens_per_ms: 17.0,
                updated_ms: 0.0,
                ..Default::default()
            },
        );
        let d = router.route(&prompt, 0.0, 3000.0).expect("route");
        assert_eq!(d.prefill, WorkerId(1));
        assert_eq!(d.prefix_hit_tokens, 4000);
    }

    #[test]
    fn a_request_that_cannot_make_its_budget_is_flagged_at_routing_time() {
        let router = Router::new(registry(&[(1, 200_000, 17.0)]), RouterTuning::default());
        let d = router.route(&vec![1u32; 4000], 0.0, 3000.0).expect("route");
        assert!(d.predicted_late, "predicted {} ms", d.predicted_ttft_ms);
    }

    #[test]
    fn an_empty_or_stale_pool_is_an_error_not_a_default() {
        let mut r = WorkerRegistry::new(1_000.0);
        r.register(worker(1, WorkerRole::Prefill, 0, 17.0));
        let router = Router::new(r, RouterTuning::default());
        assert_eq!(
            router.route(&[1, 2, 3], 0.0, 3000.0),
            Err(RoutingError::NoDecodeWorker)
        );
        // Heartbeat older than stale_after_ms: the worker disappears.
        assert_eq!(
            router.route(&[1, 2, 3], 10_000.0, 3000.0),
            Err(RoutingError::NoPrefillWorker)
        );
    }

    #[test]
    fn the_decode_worker_with_the_most_headroom_wins() {
        let mut r = WorkerRegistry::new(5_000.0);
        r.register(worker(1, WorkerRole::Prefill, 0, 17.0));
        let mut busy = worker(10, WorkerRole::Decode, 0, 0.0);
        busy.load.decode_concurrency = 60;
        let mut idle = worker(11, WorkerRole::Decode, 0, 0.0);
        idle.load.decode_concurrency = 10;
        r.register(busy);
        r.register(idle);
        let router = Router::new(r, RouterTuning::default());
        assert_eq!(
            router.route(&[1, 2, 3], 0.0, 3000.0).expect("route").decode,
            WorkerId(11)
        );
    }
}
