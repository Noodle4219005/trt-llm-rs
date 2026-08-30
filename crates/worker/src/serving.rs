//! The serving path: Rust frontend and router in front of real engine processes.
//!
//! This is the deployment ADR 0034 settled on, and it is deliberately *not*
//! [`crate::deployment::Deployment`]:
//!
//! ```text
//!   frontend (axum) -> router -> one HTTP client per worker
//!                                      |  POST /generate, SSE back
//!                                      v
//!   Python trtllm-serve x N  (PyExecutor loop untouched;
//!                             self.scheduler = RustScheduler)
//! ```
//!
//! `Deployment` drives the engine a step at a time through [`trtllm_engine::Engine`]
//! (`decode_step(now)`), which assumes **Rust owns the loop**. Under ADR 0034
//! TensorRT-LLM owns the loop, so serving needs a *request*-granularity
//! interface instead: submit, then stream tokens until a terminal event. That is
//! exactly [`trtllm_wire`], and it is already exercised -- 13/13 integration
//! tests plus a real 320-request GPU run on 2026-08-30.
//!
//! `Deployment` stays, because `crates/sim` needs the step-granularity `Engine`
//! to advance simulated time. Two interfaces, two jobs, no shared pretence.
//!
//! **Load is measured here, not reported by the worker.** `WorkerLoad`'s own
//! documentation says a load signal that needs a special code path is a load
//! signal that goes stale, and the two numbers the router actually needs --
//! in-flight count and inter-token interval -- are both observable from this
//! side of the wire with no cooperation from the worker at all.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::sync::mpsc;
use trtllm_core::{Error, Millis, Request, RequestId, Result, TokenId, WorkerId};
use trtllm_frontend::api::{Completions, GenerateRequest, StreamChunk};
use trtllm_router::policy::{Router, RouterTuning};
use trtllm_router::registry::{WorkerLoad, WorkerRegistry, WorkerRole, WorkerState};
use trtllm_wire::{DynamoAdapter, HttpTransportFactory, StreamOutput};

use crate::tokenizer::Tokenizer;

const CHANNEL_DEPTH: usize = 256;

/// One engine process the router can send to.
///
/// `slot` is the index of the process, distinct from the registry ids that
/// point at it: an aggregated worker occupies *two* registry entries (see
/// [`ServingDeployment::aggregated`]) and both must report the same load,
/// because they are the same process.
pub struct ServingWorker {
    pub slot: usize,
    pub endpoint: String,
    adapter: Arc<DynamoAdapter<HttpTransportFactory>>,
}

impl ServingWorker {
    pub fn new(slot: usize, endpoint: impl Into<String>) -> Self {
        let endpoint = endpoint.into();
        let adapter = Arc::new(DynamoAdapter::new(HttpTransportFactory::new(&endpoint)));
        Self {
            slot,
            endpoint,
            adapter,
        }
    }
}

/// What this side of the wire can see about a worker without asking it.
#[derive(Debug, Default, Clone, Copy)]
struct Observed {
    inflight: u32,
    /// Exponentially weighted mean inter-token interval, milliseconds. This is
    /// the worker's decode step time as the network sees it: with one sequence
    /// per stream it is the step, and with many it is the step divided by the
    /// batch, which is what the router's headroom estimate wants anyway.
    itl_ms: f64,
    samples: u64,
}

pub struct ServingDeployment {
    router: Mutex<Router>,
    /// Registry id -> process. Two ids may share one process.
    workers: HashMap<WorkerId, Arc<ServingWorker>>,
    /// Keyed by process, not by registry id: an aggregated worker has two ids
    /// and one queue, and giving each id its own counters would halve the
    /// apparent load of every worker.
    observed: Arc<Mutex<HashMap<usize, Observed>>>,
    tokenizer: Arc<dyn Tokenizer>,
    model_name: String,
    started: Instant,
    ttft_budget_ms: f64,
    decode_cap_hint: f64,
    next_id: AtomicU64,
}

impl ServingDeployment {
    /// Build an **aggregated** deployment: every worker serves both phases, so
    /// each is registered under both roles and the router's prefill/decode pick
    /// collapses onto one process. That is the shape NVIDIA's own H200 recipe
    /// uses for this model (`trtllm/agg/hopper/deploy.yaml`: TP4 + EP4, four
    /// replicas), and it keeps KV in the process that computed it -- no
    /// transceiver, no `/tmp` lock, none of the disagg failure modes.
    pub fn aggregated(
        endpoints: &[String],
        model_name: impl Into<String>,
        tokenizer: Arc<dyn Tokenizer>,
        tuning: RouterTuning,
        ttft_budget_ms: f64,
        decode_cap_hint: f64,
        stale_after_ms: f64,
    ) -> Result<Self> {
        if endpoints.is_empty() {
            return Err(Error::Engine("no worker endpoints given".into()));
        }
        let mut registry = WorkerRegistry::new(stale_after_ms);
        let mut workers = HashMap::new();
        let mut observed = HashMap::new();

        for (slot, endpoint) in endpoints.iter().enumerate() {
            // Two registry ids per process. `WorkerRegistry` is keyed by
            // WorkerId alone, so registering one id twice under two roles
            // silently overwrites the first -- job 314823 failed its smoke with
            // "no healthy prefill worker" for exactly that reason, having
            // registered a prefill entry and then replaced it with a decode one.
            let worker = Arc::new(ServingWorker::new(slot, endpoint));
            for (offset, role) in [(0u32, WorkerRole::Prefill), (1u32, WorkerRole::Decode)] {
                let id = WorkerId(slot as u32 * 2 + offset);
                registry.register(WorkerState {
                    id,
                    role,
                    endpoint: endpoint.clone(),
                    tensor_parallel: 1,
                    healthy: true,
                    load: WorkerLoad {
                        decode_cap: decode_cap_hint,
                        ..WorkerLoad::default()
                    },
                });
                workers.insert(id, worker.clone());
            }
            observed.insert(slot, Observed::default());
        }

        Ok(Self {
            router: Mutex::new(Router::new(registry, tuning)),
            workers,
            observed: Arc::new(Mutex::new(observed)),
            tokenizer,
            model_name: model_name.into(),
            started: Instant::now(),
            ttft_budget_ms,
            decode_cap_hint,
            next_id: AtomicU64::new(0),
        })
    }

    fn now_ms(&self) -> Millis {
        self.started.elapsed().as_secs_f64() * 1000.0
    }

    /// Push what we have observed back into the registry so the next routing
    /// decision uses it. Called on every request boundary rather than on a
    /// timer: a load signal is only useful at the moment a decision is made.
    fn refresh_load(&self, now: Millis) {
        let observed = self.observed.lock().clone();
        let mut router = self.router.lock();
        for (&id, worker) in &self.workers {
            let obs = observed.get(&worker.slot).copied().unwrap_or_default();
            let load = WorkerLoad {
                queue_depth: obs.inflight,
                decode_concurrency: obs.inflight,
                decode_cap: self.decode_cap_hint,
                step_ms: obs.itl_ms,
                updated_ms: now,
                ..WorkerLoad::default()
            };
            router.registry.update_load(id, load);
        }
    }
}

#[async_trait]
impl Completions for ServingDeployment {
    async fn generate(&self, req: GenerateRequest) -> Result<mpsc::Receiver<StreamChunk>> {
        let now = self.now_ms();
        self.refresh_load(now);

        let decision = {
            let router = self.router.lock();
            router
                .route(&req.prompt, now, self.ttft_budget_ms)
                .map_err(|e| Error::Rejected {
                    id: req.id,
                    reason: e.as_str().to_string(),
                })?
        };
        // Aggregated: one process serves the whole request. The decode pick is
        // the one that matters, because decode is what binds req/s.
        let worker = self
            .workers
            .get(&decision.decode)
            .ok_or_else(|| {
                Error::Engine(format!("routed to unknown worker {:?}", decision.decode))
            })?
            .clone();

        let core_request = Request {
            id: RequestId(self.next_id.fetch_add(1, Ordering::Relaxed)),
            prompt: req.prompt.clone(),
            params: req.params.clone(),
            arrival_ms: req.arrival_ms,
            ttft_deadline_ms: req.arrival_ms + self.ttft_budget_ms,
            prefill_worker: None,
            decode_worker: None,
        };

        let mut stream = worker
            .adapter
            .start(&core_request)
            .await
            .map_err(|e| Error::Engine(format!("transport: {e}")))?;

        let (tx, rx) = mpsc::channel::<StreamChunk>(CHANNEL_DEPTH);
        let observed = self.observed.clone();
        let started = self.started;
        let slot = worker.slot;

        {
            let mut o = observed.lock();
            o.entry(slot).or_default().inflight += 1;
        }

        tokio::spawn(async move {
            let now_ms = || started.elapsed().as_secs_f64() * 1000.0;
            let mut last_token_ms: Option<f64> = None;

            loop {
                let at = now_ms();
                match stream.next_at(at).await {
                    Ok(Some(StreamOutput::Token { token, text })) => {
                        if let Some(prev) = last_token_ms {
                            let gap = at - prev;
                            let mut o = observed.lock();
                            let e = o.entry(slot).or_default();
                            // EWMA with a short memory: the router wants what
                            // the worker is doing now, not its average since
                            // startup.
                            e.itl_ms = if e.samples == 0 {
                                gap
                            } else {
                                0.8 * e.itl_ms + 0.2 * gap
                            };
                            e.samples += 1;
                        }
                        last_token_ms = Some(at);
                        if tx.send(StreamChunk::Token { token, text }).await.is_err() {
                            // Receiver gone: the client hung up. Dropping
                            // `stream` cancels exactly once (trtllm_wire).
                            break;
                        }
                    }
                    Ok(Some(StreamOutput::Terminal { finish_reason })) => {
                        let _ = tx
                            .send(StreamChunk::Done {
                                finish_reason: leak_reason(&finish_reason),
                            })
                            .await;
                        break;
                    }
                    Ok(None) => break,
                    Err(e) => {
                        let _ = tx
                            .send(StreamChunk::Error {
                                message: e.to_string(),
                            })
                            .await;
                        break;
                    }
                }
            }

            let mut o = observed.lock();
            let e = o.entry(slot).or_default();
            e.inflight = e.inflight.saturating_sub(1);
            // One hop from the client. Comparing this against the engine's own
            // per-token interval brackets where a missing millisecond lives:
            // above this point (engine or worker HTTP) or below it (frontend to
            // client). Job 315414 needs that bracket -- engine 14.16 ms/token,
            // client 39.17, and three hypotheses about the gap already refuted.
            if e.samples > 0 && e.samples % 200 == 0 {
                tracing::info!(
                    worker = slot,
                    itl_ms = e.itl_ms,
                    samples = e.samples,
                    inflight = e.inflight,
                    "control-plane observed inter-token interval"
                );
            }
        });

        Ok(rx)
    }

    fn model_name(&self) -> String {
        self.model_name.clone()
    }

    fn encode(&self, text: &str) -> Vec<TokenId> {
        self.tokenizer.encode(text)
    }

    fn decode(&self, tokens: &[TokenId]) -> String {
        self.tokenizer.decode(tokens)
    }
}

/// `StreamChunk::Done` wants a `&'static str`, and the wire gives an owned
/// String. The set of reasons is closed and small, so map it rather than leak:
/// an unknown reason is mapped to "error" instead of being invented into a
/// static, because a finish reason we do not recognise is not a normal stop.
fn leak_reason(reason: &str) -> &'static str {
    match reason {
        "eos" => "eos",
        "stop" => "stop",
        "length" => "length",
        "cancelled" => "cancelled",
        "timeout" => "timeout",
        "content_filter" => "content_filter",
        _ => "error",
    }
}
