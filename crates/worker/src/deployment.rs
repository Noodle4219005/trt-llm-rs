use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::sync::mpsc;
use trtllm_core::config::Config;
use trtllm_core::{Error, Millis, RequestId, Result, TokenId, WorkerId};
use trtllm_engine::Engine;
use trtllm_frontend::{Completions, GenerateRequest, StreamChunk};
use trtllm_router::{Router, RouterTuning, WorkerLoad, WorkerRegistry, WorkerRole, WorkerState};
use trtllm_sched::prefill::PrefillTuning;
use trtllm_sched::{DecodeScheduler, ItlController};

use crate::decode_worker::{DecodeAdmit, DecodeEvent, DecodeWorker};
use crate::prefill_worker::{PrefillComplete, PrefillJob, PrefillWorker};
use crate::tokenizer::{SyntheticTokenizer, Tokenizer};

const DECODE_ID_BASE: u32 = 1000;
const CHANNEL_DEPTH: usize = 4096;

struct Inflight {
    tx: mpsc::Sender<StreamChunk>,
    decode_worker: usize,
    max_tokens: u32,
}

/// A whole deployment - router, prefill workers, decode workers - in one
/// process.
pub struct Deployment {
    cfg: Config,
    tokenizer: Arc<dyn Tokenizer>,
    router: Mutex<Router>,
    prefill: Vec<Arc<PrefillWorker>>,
    decode: Vec<Arc<DecodeWorker>>,
    prefill_tx: Vec<mpsc::Sender<PrefillJob>>,
    decode_tx: Vec<mpsc::Sender<DecodeAdmit>>,
    inflight: Mutex<HashMap<RequestId, Inflight>>,
    epoch: Instant,
}

/// Keeps the background tasks alive; dropping it shuts the deployment down.
pub struct DeploymentHandle {
    pub deployment: Arc<Deployment>,
    tasks: Vec<tokio::task::JoinHandle<()>>,
}

impl DeploymentHandle {
    pub async fn shutdown(mut self) {
        for t in self.tasks.drain(..) {
            t.abort();
        }
    }
}

impl Deployment {
    /// Build and start a deployment from already-constructed engines.
    ///
    /// The engines are the only thing that differs between the GPU-free demo
    /// and production. Everything below this line - queueing, batching,
    /// admission, routing - is identical in both.
    pub fn spawn(
        cfg: Config,
        prefill_engines: Vec<Arc<dyn Engine>>,
        decode_engines: Vec<Arc<dyn Engine>>,
        tokenizer: Option<Arc<dyn Tokenizer>>,
    ) -> Result<DeploymentHandle> {
        cfg.validate()?;
        if prefill_engines.is_empty() || decode_engines.is_empty() {
            return Err(Error::Config(
                "need at least one prefill and one decode engine".into(),
            ));
        }
        let epoch = Instant::now();
        let tokenizer = tokenizer.unwrap_or_else(|| Arc::new(SyntheticTokenizer));

        let cal = cfg.calibration.prefill;
        let per_worker_rate = cal.tok_s_per_gpu_at_tp(cfg.topology.prefill_tp)
            * f64::from(cfg.topology.prefill_tp)
            / 1000.0;

        let mut prefill = Vec::new();
        let mut prefill_tx = Vec::new();
        let mut tasks = Vec::new();
        let (pdone_tx, pdone_rx) = mpsc::channel::<PrefillComplete>(CHANNEL_DEPTH);

        for (i, engine) in prefill_engines.into_iter().enumerate() {
            let w = Arc::new(PrefillWorker::new(
                WorkerId(i as u32),
                engine,
                PrefillTuning {
                    chunk_tokens: cfg.scheduler.chunked_prefill_tokens as usize,
                    max_seqs: cfg.scheduler.max_prefill_seqs as usize,
                    policy: cfg.scheduler.prefill_policy,
                    demote_hopeless: cfg.scheduler.demote_hopeless,
                    tokens_per_ms: per_worker_rate,
                    rate_alpha: 0.2,
                },
                cfg.kv.num_blocks,
                cfg.kv.block_size,
                cfg.kv.admission_watermark,
                epoch,
            ));
            let (tx, rx) = mpsc::channel::<PrefillJob>(CHANNEL_DEPTH);
            tasks.push(tokio::spawn(w.clone().run(rx, pdone_tx.clone())));
            prefill.push(w);
            prefill_tx.push(tx);
        }
        drop(pdone_tx);

        let itl_target = cfg.slo.itl_ms * cfg.scheduler.itl_safety;
        let initial_cap =
            cfg.calibration.decode.concurrency_per_gpu * f64::from(cfg.topology.decode_tp);
        let mut decode = Vec::new();
        let mut decode_tx = Vec::new();
        let (devent_tx, devent_rx) = mpsc::channel::<DecodeEvent>(CHANNEL_DEPTH);

        for (j, engine) in decode_engines.into_iter().enumerate() {
            let mut sched = DecodeScheduler::new(
                cfg.slo.itl_ms,
                ItlController::new(
                    itl_target,
                    initial_cap,
                    8.0,
                    f64::from(cfg.scheduler.max_decode_seqs),
                ),
            );
            // Speculation emits more than one token per accepted step, and
            // `remaining_tokens` and `tolerable_itl_ms` both divide by the
            // count. Setting it here keeps the scheduler and
            // `speculative_config` describing the same engine.
            if cfg.engine.speculation.enabled {
                sched.set_tokens_per_step(1 + cfg.engine.speculation.draft_tokens);
            }
            let w = Arc::new(DecodeWorker::new(
                WorkerId(DECODE_ID_BASE + j as u32),
                engine,
                sched,
                cfg.kv.num_blocks,
                cfg.kv.block_size,
                cfg.kv.admission_watermark,
                epoch,
            ));
            let (tx, rx) = mpsc::channel::<DecodeAdmit>(CHANNEL_DEPTH);
            tasks.push(tokio::spawn(w.clone().run(rx, devent_tx.clone())));
            decode.push(w);
            decode_tx.push(tx);
        }
        drop(devent_tx);

        let mut registry = WorkerRegistry::new(60_000.0);
        for w in &prefill {
            registry.register(WorkerState {
                id: w.id,
                role: WorkerRole::Prefill,
                endpoint: format!("inproc://prefill/{}", w.id),
                tensor_parallel: cfg.topology.prefill_tp,
                healthy: true,
                load: WorkerLoad {
                    tokens_per_ms: per_worker_rate,
                    updated_ms: 0.0,
                    ..Default::default()
                },
            });
        }
        for w in &decode {
            registry.register(WorkerState {
                id: w.id,
                role: WorkerRole::Decode,
                endpoint: format!("inproc://decode/{}", w.id),
                tensor_parallel: cfg.topology.decode_tp,
                healthy: true,
                load: WorkerLoad {
                    decode_cap: initial_cap,
                    updated_ms: 0.0,
                    ..Default::default()
                },
            });
        }
        let router = Router::new(
            registry,
            RouterTuning {
                default_tokens_per_ms: per_worker_rate,
                ..Default::default()
            },
        );

        let dep = Arc::new(Self {
            cfg,
            tokenizer,
            router: Mutex::new(router),
            prefill,
            decode,
            prefill_tx,
            decode_tx,
            inflight: Mutex::new(HashMap::new()),
            epoch,
        });

        tasks.push(tokio::spawn(pump_prefill(dep.clone(), pdone_rx)));
        tasks.push(tokio::spawn(pump_decode(dep.clone(), devent_rx)));
        tasks.push(tokio::spawn(sync_loads(dep.clone())));

        Ok(DeploymentHandle {
            deployment: dep,
            tasks,
        })
    }

    pub fn now_ms(&self) -> Millis {
        self.epoch.elapsed().as_secs_f64() * 1000.0
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// A snapshot for `/metrics`-style reporting.
    pub fn snapshot(&self) -> Vec<(String, f64)> {
        let mut v = Vec::new();
        for w in &self.prefill {
            v.push((
                format!("prefill{}_queue_depth", w.id),
                w.queue_depth() as f64,
            ));
            v.push((
                format!("prefill{}_queued_tokens", w.id),
                w.queued_tokens() as f64,
            ));
            v.push((format!("prefill{}_tokens_per_ms", w.id), w.rate()));
        }
        for w in &self.decode {
            v.push((
                format!("decode{}_concurrency", w.id),
                w.concurrency() as f64,
            ));
            v.push((format!("decode{}_cap", w.id), w.cap()));
            v.push((format!("decode{}_step_ms", w.id), w.observed_step_ms()));
            v.push((format!("decode{}_kv_utilisation", w.id), w.kv_utilisation()));
            v.push((format!("decode{}_refusals", w.id), w.refusals() as f64));
        }
        v.push(("inflight".into(), self.inflight.lock().len() as f64));
        v
    }
}

async fn pump_prefill(dep: Arc<Deployment>, mut rx: mpsc::Receiver<PrefillComplete>) {
    while let Some(done) = rx.recv().await {
        let (tx, worker, max_tokens) = {
            let inflight = dep.inflight.lock();
            match inflight.get(&done.id) {
                Some(s) => (s.tx.clone(), s.decode_worker, s.max_tokens),
                None => continue,
            }
        };
        // Time to first token stops here: the prefill worker sampled it.
        let text = dep.tokenizer.decode(&[done.first_token]);
        if tx
            .send(StreamChunk::Token {
                token: done.first_token,
                text,
            })
            .await
            .is_err()
        {
            dep.inflight.lock().remove(&done.id);
            continue;
        }
        if max_tokens <= 1 {
            let _ = tx
                .send(StreamChunk::Done {
                    finish_reason: "length",
                })
                .await;
            dep.inflight.lock().remove(&done.id);
            continue;
        }
        let admit = DecodeAdmit {
            id: done.id,
            first_token: done.first_token,
            prompt_len: done.prompt_len,
            max_tokens,
            first_token_ms: done.finished_at_ms,
        };
        if dep.decode_tx[worker].send(admit).await.is_err() {
            let _ = tx
                .send(StreamChunk::Error {
                    message: "decode worker gone".into(),
                })
                .await;
            dep.inflight.lock().remove(&done.id);
        }
    }
}

async fn pump_decode(dep: Arc<Deployment>, mut rx: mpsc::Receiver<DecodeEvent>) {
    while let Some(ev) = rx.recv().await {
        match ev {
            DecodeEvent::Token { id, token } => {
                let tx = { dep.inflight.lock().get(&id).map(|s| s.tx.clone()) };
                if let Some(tx) = tx {
                    let text = dep.tokenizer.decode(&[token]);
                    if tx.send(StreamChunk::Token { token, text }).await.is_err() {
                        dep.inflight.lock().remove(&id);
                    }
                }
            }
            DecodeEvent::Finished { id, .. } => {
                let entry = dep.inflight.lock().remove(&id);
                if let Some(s) = entry {
                    let _ =
                        s.tx.send(StreamChunk::Done {
                            finish_reason: "length",
                        })
                        .await;
                }
            }
        }
    }
}

/// Push worker load into the registry so routing decisions are made on fresh
/// numbers. A stale load signal routes everything to whichever worker last
/// reported, which looks exactly like a broken load balancer.
async fn sync_loads(dep: Arc<Deployment>) {
    let period = std::time::Duration::from_millis(20);
    loop {
        tokio::time::sleep(period).await;
        let now = dep.now_ms();
        let mut router = dep.router.lock();
        for w in &dep.prefill {
            router.registry.update_load(
                w.id,
                WorkerLoad {
                    queued_tokens: w.queued_tokens() as u64,
                    queue_depth: w.queue_depth() as u32,
                    tokens_per_ms: w.rate(),
                    updated_ms: now,
                    ..Default::default()
                },
            );
        }
        for w in &dep.decode {
            router.registry.update_load(
                w.id,
                WorkerLoad {
                    decode_concurrency: w.concurrency() as u32,
                    decode_cap: w.cap(),
                    kv_utilisation: w.kv_utilisation(),
                    step_ms: w.observed_step_ms(),
                    updated_ms: now,
                    ..Default::default()
                },
            );
        }
    }
}

#[async_trait]
impl Completions for Deployment {
    async fn generate(&self, req: GenerateRequest) -> Result<mpsc::Receiver<StreamChunk>> {
        let now = self.now_ms();
        let decision = {
            let router = self.router.lock();
            router
                .route(&req.prompt, now, self.cfg.slo.ttft_ms)
                .map_err(|e| Error::Rejected {
                    id: req.id,
                    reason: e.as_str().to_string(),
                })?
        };
        let pw = decision.prefill.0 as usize;
        let dw = (decision.decode.0 - DECODE_ID_BASE) as usize;

        let (tx, rx) = mpsc::channel::<StreamChunk>(CHANNEL_DEPTH);
        self.inflight.lock().insert(
            req.id,
            Inflight {
                tx,
                decode_worker: dw,
                max_tokens: req.params.max_tokens,
            },
        );

        let job = PrefillJob {
            id: req.id,
            prompt: req.prompt,
            arrival_ms: req.arrival_ms,
            deadline_ms: req.arrival_ms + self.cfg.slo.ttft_ms,
        };
        self.prefill_tx[pw]
            .send(job)
            .await
            .map_err(|_| Error::Engine("prefill worker gone".into()))?;
        Ok(rx)
    }

    fn model_name(&self) -> String {
        self.cfg.model.name.clone()
    }

    fn encode(&self, text: &str) -> Vec<TokenId> {
        self.tokenizer.encode(text)
    }

    fn decode(&self, tokens: &[TokenId]) -> String {
        self.tokenizer.decode(tokens)
    }
}
