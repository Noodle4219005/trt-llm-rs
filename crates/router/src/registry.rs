use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use trtllm_core::{Millis, Phase, WorkerId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerRole {
    Prefill,
    Decode,
}

impl From<Phase> for WorkerRole {
    fn from(p: Phase) -> Self {
        match p {
            Phase::Prefill => WorkerRole::Prefill,
            Phase::Decode => WorkerRole::Decode,
        }
    }
}

/// What a worker reports about itself. Everything here is observable at the
/// worker with no extra instrumentation, which is deliberate: a load signal
/// that needs a special code path is a load signal that goes stale.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct WorkerLoad {
    /// Prefill: tokens still queued. Decode: unused.
    pub queued_tokens: u64,
    pub queue_depth: u32,
    /// Decode: sequences currently in the batch.
    pub decode_concurrency: u32,
    /// Decode: the admission cap the worker's ITL controller has settled on.
    pub decode_cap: f64,
    /// Fraction of the KV pool in use, 0..1.
    pub kv_utilisation: f64,
    /// Measured token rate, tokens/ms. Prefill only.
    pub tokens_per_ms: f64,
    /// Measured decode step latency, milliseconds. Decode only.
    pub step_ms: f64,
    pub updated_ms: Millis,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkerState {
    pub id: WorkerId,
    pub role: WorkerRole,
    pub endpoint: String,
    pub tensor_parallel: u32,
    pub healthy: bool,
    pub load: WorkerLoad,
}

impl WorkerState {
    /// Free decode slots under the worker's own admission cap.
    pub fn decode_headroom(&self) -> f64 {
        (self.load.decode_cap - f64::from(self.load.decode_concurrency)).max(0.0)
    }
}

#[derive(Debug, Default)]
pub struct WorkerRegistry {
    workers: HashMap<WorkerId, WorkerState>,
    /// Heartbeats older than this mark a worker unhealthy.
    pub stale_after_ms: f64,
}

impl WorkerRegistry {
    pub fn new(stale_after_ms: f64) -> Self {
        Self {
            workers: HashMap::new(),
            stale_after_ms,
        }
    }

    pub fn register(&mut self, w: WorkerState) {
        self.workers.insert(w.id, w);
    }

    pub fn deregister(&mut self, id: WorkerId) -> Option<WorkerState> {
        self.workers.remove(&id)
    }

    pub fn get(&self, id: WorkerId) -> Option<&WorkerState> {
        self.workers.get(&id)
    }

    pub fn update_load(&mut self, id: WorkerId, load: WorkerLoad) {
        if let Some(w) = self.workers.get_mut(&id) {
            w.load = load;
            w.healthy = true;
        }
    }

    pub fn len(&self) -> usize {
        self.workers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.workers.is_empty()
    }

    /// Healthy workers of one role, in deterministic id order so routing is
    /// reproducible across runs.
    pub fn live(&self, role: WorkerRole, now: Millis) -> Vec<&WorkerState> {
        let mut v: Vec<&WorkerState> = self
            .workers
            .values()
            .filter(|w| {
                w.role == role && w.healthy && now - w.load.updated_ms <= self.stale_after_ms
            })
            .collect();
        v.sort_by_key(|w| w.id);
        v
    }

    pub fn all(&self) -> impl Iterator<Item = &WorkerState> {
        self.workers.values()
    }

    /// Total prefill token rate across the healthy pool - the denominator the
    /// prefill scheduler's deadline arithmetic needs.
    pub fn prefill_rate(&self, now: Millis) -> f64 {
        self.live(WorkerRole::Prefill, now)
            .iter()
            .map(|w| w.load.tokens_per_ms)
            .sum()
    }
}
