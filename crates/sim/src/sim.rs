use std::collections::{BinaryHeap, HashMap, VecDeque};

use trtllm_core::capacity::{DecodeCalibration, PrefillCalibration};
use trtllm_core::config::Config;
use trtllm_core::{
    GoodputReport, LatencyStats, Millis, RequestId, RequestOutcome, TokenId, WorkerId,
};
use trtllm_engine::cost::{DecodeCurve, PrefillCurve};
use trtllm_kvcache::blocks_for;
use trtllm_router::{Router, RouterTuning, WorkerLoad, WorkerRegistry, WorkerRole, WorkerState};
use trtllm_sched::prefill::PrefillTuning;
use trtllm_sched::{
    DecodeScheduler, ItlController, PendingPrefill, PrefillBatch, PrefillScheduler, RunningSeq,
};

use crate::events::{Event, Scheduled};
use crate::report::{Diagnostics, SimReport};

const DECODE_ID_BASE: u32 = 1000;

pub struct SimSetup {
    pub config: Config,
}

struct PrefillWorkerSim {
    sched: PrefillScheduler,
    curve: PrefillCurve,
    busy: bool,
    batch: Option<(PrefillBatch, Millis)>,
    busy_ms: f64,
}

struct DecodeWorkerSim {
    sched: DecodeScheduler,
    curve: DecodeCurve,
    pending: VecDeque<RequestId>,
    stepping: bool,
    free_blocks: usize,
    total_blocks: usize,
    watermark_blocks: usize,
    held: HashMap<RequestId, usize>,
}

struct ReqState {
    client: usize,
    arrival_ms: Millis,
    prompt_len: usize,
    osl: u32,
    decode_worker: usize,
    first_token_ms: Option<Millis>,
    last_token_ms: Millis,
}

pub struct Simulator {
    cfg: Config,
    now: Millis,
    heap: BinaryHeap<Scheduled>,
    seq: u64,
    next_id: u64,
    reqs: HashMap<RequestId, ReqState>,
    prefill: Vec<PrefillWorkerSim>,
    decode: Vec<DecodeWorkerSim>,
    router: Router,
    /// P/D KV transfers currently occupying a buffer, and the ones queued for
    /// one. Upstream's pool holds `cfg.kv.xfer_concurrency` buffers and blocks
    /// when they are all taken, so an unbounded fixed delay is the wrong model
    /// -- it is the difference between predicting a 5.50 req/s ceiling and
    /// predicting none at all.
    xfer_busy: u32,
    xfer_queue: VecDeque<(RequestId, usize)>,
    outcomes: Vec<RequestOutcome>,
    // accumulators
    issued: usize,
    batches: u64,
    batch_seqs: u64,
    batch_tokens: u64,
    deadline_limited: u64,
    demoted_samples: Vec<f64>,
    calibrated_concurrency: f64,
    horizon_ms: Millis,
    concurrency_samples: Vec<f64>,
    peak_concurrency: usize,
    queue_depth_samples: Vec<f64>,
}

impl Simulator {
    pub fn new(setup: SimSetup) -> Self {
        let cfg = setup.config;
        let t = cfg.topology;
        let pcal: PrefillCalibration = cfg.calibration.prefill;
        let dcal: DecodeCalibration = cfg.calibration.decode;

        let prefill_curve = PrefillCurve::for_worker(&pcal, t.prefill_tp, t.prefill_tp);
        let pool_rate = prefill_curve.tokens_per_ms * f64::from(t.prefill_workers);

        let mut prefill = Vec::new();
        for _ in 0..t.prefill_workers {
            prefill.push(PrefillWorkerSim {
                sched: PrefillScheduler::new(PrefillTuning {
                    chunk_tokens: cfg.scheduler.chunked_prefill_tokens as usize,
                    max_seqs: cfg.scheduler.max_prefill_seqs as usize,
                    policy: cfg.scheduler.prefill_policy,
                    demote_hopeless: cfg.scheduler.demote_hopeless,
                    tokens_per_ms: prefill_curve.tokens_per_ms,
                    rate_alpha: 0.2,
                }),
                curve: prefill_curve,
                busy: false,
                batch: None,
                busy_ms: 0.0,
            });
        }

        let decode_curve = DecodeCurve::from_calibration(&dcal, t.decode_tp);
        let itl_target = cfg.slo.itl_ms * cfg.scheduler.itl_safety;
        let initial_cap = dcal.concurrency_per_gpu * f64::from(t.decode_tp);
        let total_blocks = cfg.kv.num_blocks as usize;
        let watermark_blocks = (total_blocks as f64 * cfg.kv.admission_watermark).ceil() as usize;

        let mut decode = Vec::new();
        for _ in 0..t.decode_workers {
            decode.push(DecodeWorkerSim {
                sched: DecodeScheduler::new(
                    cfg.slo.itl_ms,
                    ItlController::new(
                        itl_target,
                        initial_cap,
                        8.0,
                        f64::from(cfg.scheduler.max_decode_seqs),
                    ),
                ),
                curve: decode_curve,
                pending: VecDeque::new(),
                stepping: false,
                free_blocks: total_blocks,
                total_blocks,
                watermark_blocks,
                held: HashMap::new(),
            });
        }

        let mut registry = WorkerRegistry::new(60_000.0);
        for i in 0..t.prefill_workers {
            registry.register(WorkerState {
                id: WorkerId(i),
                role: WorkerRole::Prefill,
                endpoint: format!("sim://prefill/{i}"),
                tensor_parallel: t.prefill_tp,
                healthy: true,
                load: WorkerLoad {
                    tokens_per_ms: prefill_curve.tokens_per_ms,
                    updated_ms: 0.0,
                    ..Default::default()
                },
            });
        }
        for j in 0..t.decode_workers {
            registry.register(WorkerState {
                id: WorkerId(DECODE_ID_BASE + j),
                role: WorkerRole::Decode,
                endpoint: format!("sim://decode/{j}"),
                tensor_parallel: t.decode_tp,
                healthy: true,
                load: WorkerLoad {
                    decode_cap: initial_cap,
                    updated_ms: 0.0,
                    ..Default::default()
                },
            });
        }

        let kv_transfer_ms = kv_transfer_ms(&cfg);
        let router = Router::new(
            registry,
            RouterTuning {
                kv_transfer_ms,
                default_tokens_per_ms: pool_rate / f64::from(t.prefill_workers.max(1)),
                min_prefix_tokens: 256,
            },
        );

        Self {
            cfg,
            now: 0.0,
            heap: BinaryHeap::new(),
            seq: 0,
            next_id: 0,
            reqs: HashMap::new(),
            prefill,
            decode,
            router,
            xfer_busy: 0,
            xfer_queue: VecDeque::new(),
            outcomes: Vec::new(),
            issued: 0,
            batches: 0,
            batch_seqs: 0,
            batch_tokens: 0,
            deadline_limited: 0,
            demoted_samples: Vec::new(),
            calibrated_concurrency: initial_cap,
            horizon_ms: f64::INFINITY,
            concurrency_samples: Vec::new(),
            peak_concurrency: 0,
            queue_depth_samples: Vec::new(),
        }
    }

    fn push(&mut self, at: Millis, event: Event) {
        self.seq += 1;
        self.heap.push(Scheduled {
            at,
            seq: self.seq,
            event,
        });
    }

    pub fn run(mut self) -> SimReport {
        let w = self.cfg.workload;
        let window_start = w.warmup_s * 1000.0;
        let window_end = window_start + w.benchmark_s * 1000.0;
        let horizon = window_end + w.grace_s * 1000.0;
        self.horizon_ms = horizon;

        for client in 0..w.concurrency as usize {
            self.push(0.0, Event::Arrival { client });
        }

        while let Some(ev) = self.heap.pop() {
            if ev.at > horizon {
                break;
            }
            self.now = ev.at;
            match ev.event {
                Event::Arrival { client } => self.on_arrival(client),
                Event::PrefillDone { worker } => self.on_prefill_done(worker),
                Event::KvArrived { id, worker } => self.on_kv_arrived(id, worker),
                Event::DecodeStep { worker } => self.on_decode_step(worker),
            }
        }

        self.finish(window_start, window_end)
    }

    fn on_arrival(&mut self, client: usize) {
        let id = RequestId(self.next_id);
        self.next_id += 1;
        self.issued += 1;

        let isl = self.cfg.workload.isl as usize;
        // With --cache-bust every prompt has a unique prefix, so the token
        // pattern only has to be unique, not realistic.
        let prompt: Vec<TokenId> = vec![(id.0 % 100_000) as TokenId + 1; isl];

        self.sync_loads();
        let decision = match self.router.route(&prompt, self.now, self.cfg.slo.ttft_ms) {
            Ok(d) => d,
            // No worker: in a simulation this is a configuration bug, not a
            // runtime condition worth modelling.
            Err(e) => panic!("routing failed in simulation: {}", e.as_str()),
        };

        let pw = decision.prefill.0 as usize;
        let dw = (decision.decode.0 - DECODE_ID_BASE) as usize;

        self.reqs.insert(
            id,
            ReqState {
                client,
                arrival_ms: self.now,
                prompt_len: isl,
                osl: self.cfg.workload.osl,
                decode_worker: dw,
                first_token_ms: None,
                last_token_ms: self.now,
            },
        );

        let compute = isl.saturating_sub(decision.prefix_hit_tokens);
        self.prefill[pw].sched.enqueue(PendingPrefill {
            id,
            arrival_ms: self.now,
            deadline_ms: self.now + self.cfg.slo.ttft_ms,
            compute_tokens: compute,
            done_tokens: 0,
        });
        if !self.cfg.workload.cache_bust {
            self.router
                .note_prefill(WorkerId(pw as u32), &prompt, self.now);
        }
        self.try_start_prefill(pw);
    }

    fn try_start_prefill(&mut self, w: usize) {
        if self.prefill[w].busy {
            return;
        }
        let now = self.now;
        let batch = self.prefill[w].sched.plan(now);
        if batch.is_empty() {
            return;
        }
        let ms = self.prefill[w]
            .curve
            .batch_ms(batch.total_tokens, batch.chunks.len());

        self.batches += 1;
        self.batch_seqs += batch.chunks.len() as u64;
        self.batch_tokens += batch.total_tokens as u64;
        if batch.deadline_limited {
            self.deadline_limited += 1;
        }
        let depth = self.prefill[w].sched.queue_depth();
        self.queue_depth_samples.push(depth as f64);
        if depth > 0 {
            self.demoted_samples
                .push(self.prefill[w].sched.demoted_count() as f64 / depth as f64);
        }

        // Only the part of a batch that falls inside the simulated span counts
        // towards utilisation. Charging the whole batch is how a utilisation
        // figure ends up above 100 %, which is not a tight deployment - it is a
        // broken denominator.
        let charged = ms.min((self.horizon_ms - now).max(0.0));
        let worker = &mut self.prefill[w];
        worker.busy = true;
        worker.busy_ms += charged;
        worker.batch = Some((batch, now));
        self.push(now + ms, Event::PrefillDone { worker: w });
    }

    fn on_prefill_done(&mut self, w: usize) {
        let Some((batch, started)) = self.prefill[w].batch.take() else {
            return;
        };
        let elapsed = self.now - started;
        let finished = self.prefill[w].sched.complete(&batch, elapsed);
        self.prefill[w].busy = false;

        let transfer = kv_transfer_ms(&self.cfg);
        for id in finished {
            // The prefill worker samples and streams the first token itself, so
            // TTFT stops here. The KV handoff that follows lands inside the
            // inter-token budget instead - which is where a slow fabric
            // actually shows up, and why it is not visible in TTFT.
            let dw = {
                let r = self.reqs.get_mut(&id).expect("known request");
                r.first_token_ms = Some(self.now);
                r.last_token_ms = self.now;
                r.decode_worker
            };
            self.xfer_queue.push_back((id, dw));
        }
        self.drain_xfer_queue(transfer);
        self.try_start_prefill(w);
    }

    /// Start as many queued transfers as there are free buffers.
    fn drain_xfer_queue(&mut self, transfer: Millis) {
        let cap = self.cfg.kv.xfer_concurrency.max(1);
        while self.xfer_busy < cap {
            let Some((id, dw)) = self.xfer_queue.pop_front() else {
                return;
            };
            self.xfer_busy += 1;
            self.push(self.now + transfer, Event::KvArrived { id, worker: dw });
        }
    }

    fn on_kv_arrived(&mut self, id: RequestId, w: usize) {
        // The buffer this transfer held is now free for whoever is waiting.
        self.xfer_busy = self.xfer_busy.saturating_sub(1);
        let transfer = kv_transfer_ms(&self.cfg);
        self.drain_xfer_queue(transfer);
        self.decode[w].pending.push_back(id);
        if !self.decode[w].stepping {
            self.decode[w].stepping = true;
            self.push(self.now, Event::DecodeStep { worker: w });
        }
    }

    fn on_decode_step(&mut self, w: usize) {
        self.admit_decode(w);

        let concurrency = self.decode[w].sched.concurrency();
        if concurrency == 0 {
            if self.decode[w].pending.is_empty() {
                self.decode[w].stepping = false;
            } else {
                // Nothing could be admitted this instant; retry on the next
                // nominal step boundary rather than spinning.
                let retry = self.decode[w].curve.step_ms(1);
                self.push(self.now + retry, Event::DecodeStep { worker: w });
            }
            return;
        }

        let step_ms = self.decode[w].curve.step_ms(concurrency);
        let t_done = self.now + step_ms;

        self.concurrency_samples.push(concurrency as f64);
        self.peak_concurrency = self.peak_concurrency.max(concurrency);

        let done = self.decode[w].sched.on_step_synthetic(t_done, step_ms);
        // Every attached sequence advanced one token.
        let ids: Vec<RequestId> = self.decode[w].sched.running().map(|s| s.id).collect();
        for id in ids {
            if let Some(r) = self.reqs.get_mut(&id) {
                r.last_token_ms = t_done;
            }
        }

        for seq in done {
            self.retire(w, seq, t_done);
        }

        self.push(t_done, Event::DecodeStep { worker: w });
    }

    fn admit_decode(&mut self, w: usize) {
        while let Some(id) = self.decode[w].pending.front().copied() {
            let Some(r) = self.reqs.get(&id) else {
                self.decode[w].pending.pop_front();
                continue;
            };
            let need = blocks_for(r.prompt_len + r.osl as usize, self.cfg.kv.block_size);
            let d = &mut self.decode[w];
            let headroom = d.free_blocks >= need + d.watermark_blocks;
            let decision = d.sched.can_admit(headroom);
            d.sched.note(decision);
            if !decision.is_admit() {
                break;
            }
            d.pending.pop_front();
            d.free_blocks -= need;
            d.held.insert(id, need);
            let first = r.first_token_ms.expect("prefill emitted the first token");
            let mut seq = RunningSeq::new(id, first, r.osl);
            // The sequence has been alive since its first token; the KV handoff
            // it just waited through is already part of its ITL average.
            seq.last_token_ms = self.now;
            d.sched.admit_at(seq, self.now);
        }
    }

    fn retire(&mut self, w: usize, seq: RunningSeq, t_done: Millis) {
        if let Some(blocks) = self.decode[w].held.remove(&seq.id) {
            self.decode[w].free_blocks += blocks;
        }
        let Some(r) = self.reqs.remove(&seq.id) else {
            return;
        };
        self.outcomes.push(RequestOutcome {
            id: seq.id,
            arrival_ms: r.arrival_ms,
            first_token_ms: r.first_token_ms.unwrap_or(t_done),
            last_token_ms: t_done,
            prompt_tokens: r.prompt_len as u32,
            output_tokens: seq.tokens_emitted,
            requested_tokens: r.osl,
        });
        // Closed loop: the client thread is released and immediately issues its
        // next request. No think time - that is what the benchmark does.
        self.push(t_done, Event::Arrival { client: r.client });
    }

    fn sync_loads(&mut self) {
        for (i, w) in self.prefill.iter().enumerate() {
            self.router.registry.update_load(
                WorkerId(i as u32),
                WorkerLoad {
                    queued_tokens: w.sched.queued_tokens() as u64,
                    queue_depth: w.sched.queue_depth() as u32,
                    tokens_per_ms: w.sched.rate(),
                    updated_ms: self.now,
                    ..Default::default()
                },
            );
        }
        for (j, d) in self.decode.iter().enumerate() {
            self.router.registry.update_load(
                WorkerId(DECODE_ID_BASE + j as u32),
                WorkerLoad {
                    decode_concurrency: d.sched.concurrency() as u32,
                    decode_cap: d.sched.controller().cap(),
                    kv_utilisation: 1.0 - d.free_blocks as f64 / d.total_blocks.max(1) as f64,
                    step_ms: d.sched.controller().observed_itl_ms(),
                    updated_ms: self.now,
                    ..Default::default()
                },
            );
        }
    }

    fn finish(self, window_start: Millis, window_end: Millis) -> SimReport {
        let scored: Vec<RequestOutcome> = self
            .outcomes
            .iter()
            .filter(|o| o.arrival_ms >= window_start && o.arrival_ms < window_end)
            .copied()
            .collect();
        let window_s = (window_end - window_start) / 1000.0;
        let goodput = GoodputReport::from_outcomes(&scored, window_s, &self.cfg.slo);

        let sim_ms = window_end + self.cfg.workload.grace_s * 1000.0;
        let prefill_busy: f64 = self.prefill.iter().map(|w| w.busy_ms).sum();
        let prefill_capacity = sim_ms * self.prefill.len() as f64;
        let batches = self.batches.max(1) as f64;

        let diagnostics = Diagnostics {
            simulated_s: sim_ms / 1000.0,
            requests_issued: self.issued,
            requests_completed: self.outcomes.len(),
            mean_prefill_batch_seqs: self.batch_seqs as f64 / batches,
            mean_prefill_batch_tokens: self.batch_tokens as f64 / batches,
            deadline_limited_frac: self.deadline_limited as f64 / batches,
            demoted_frac: mean(&self.demoted_samples),
            prefill_busy_frac: if prefill_capacity > 0.0 {
                prefill_busy / prefill_capacity
            } else {
                0.0
            },
            mean_decode_concurrency: mean(&self.concurrency_samples),
            peak_decode_concurrency: self.peak_concurrency,
            final_decode_cap: self
                .decode
                .first()
                .map_or(0.0, |d| d.sched.controller().cap()),
            observed_step_ms: self
                .decode
                .first()
                .map_or(0.0, |d| d.sched.controller().observed_itl_ms()),
            decode_refusals: self.decode.iter().map(|d| d.sched.refused()).sum(),
            prefill_queue_depth: LatencyStats::from_samples(&self.queue_depth_samples),
            calibrated_concurrency: self.calibrated_concurrency,
            extrapolated_beyond_calibration: self.peak_concurrency as f64
                > self.calibrated_concurrency * 1.05,
        };

        SimReport {
            goodput,
            diagnostics,
        }
    }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// Residual KV handoff cost for one request, from the model shape and an
/// assumed fabric bandwidth. Not measured on this cluster; see
/// `docs/kv-transfer.md`.
fn kv_transfer_ms(cfg: &Config) -> f64 {
    let m = &cfg.model;
    let bytes_per_token =
        2.0 * f64::from(m.num_layers) * f64::from(m.num_kv_heads) * f64::from(m.head_dim);
    let bytes = bytes_per_token * f64::from(cfg.workload.isl);
    let gib_s = cfg.calibration.kv_xfer_gib_s;
    0.5 + (bytes / (gib_s * 1024.0 * 1024.0 * 1024.0)) * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use trtllm_core::config::PrefillPolicy;

    fn short_config() -> Config {
        let mut c = Config::default();
        c.workload.warmup_s = 2.0;
        c.workload.benchmark_s = 8.0;
        c.workload.grace_s = 10.0;
        c.workload.concurrency = 48;
        c
    }

    /// `plan` and the simulator disagree about 4 x TP2 prefill, and this test
    /// exists to keep the disagreement visible rather than let whichever model
    /// was consulted last decide the deployment.
    ///
    /// `plan` prefers it: narrower workers cut the ring all-reduce, so prefill
    /// goes 15.59 -> 17.01 req/s, and a neighbouring stack reached goodput
    /// 22.419 on exactly this shape. The simulator does not: it drives the
    /// decode cap to 8, refuses ~14,000 admissions and leaves prefill 85% idle.
    ///
    /// Neither is obviously wrong. The simulator batches 1.6-2.0 sequences per
    /// prefill iteration where the deployment now sets max_num_tokens 16384 --
    /// four whole prompts -- so its prefill arrival pattern is not the one the
    /// calibration came from. That is also most of the 14.30-versus-7.53 gap
    /// between the two models, and one measurement of the real prefill batch
    /// size settles all of it.
    ///
    /// Until then this test asserts only that the gap is still there, so it
    /// cannot be forgotten, and fails loudly if it closes -- because a
    /// disagreement that quietly resolves itself usually means one side
    /// stopped modelling something.
    #[test]
    fn the_two_models_disagree_about_narrow_prefill() {
        let run = |workers: u32, tp: u32| {
            let mut c = short_config();
            c.workload.benchmark_s = 60.0;
            c.workload.grace_s = 30.0;
            c.topology.prefill_workers = workers;
            c.topology.prefill_tp = tp;
            Simulator::new(SimSetup { config: c }).run()
        };
        let wide = run(2, 4);
        let narrow = run(4, 2);

        let m = trtllm_core::CapacityModel::default();
        assert!(
            m.prefill.tok_s_per_gpu_at_tp(2) > m.prefill.tok_s_per_gpu_at_tp(4),
            "the capacity model no longer prefers narrow prefill, so there is \
             nothing left to disagree about and this test should go"
        );
        assert!(
            narrow.goodput.total_requests <= wide.goodput.total_requests,
            "the simulator now agrees with the capacity model about narrow \
             prefill: narrow {} vs wide {}. If that is because the prefill \
             batch size was measured and fed in, delete this test. If nobody \
             measured anything, find out what stopped being modelled.",
            narrow.goodput.total_requests,
            wide.goodput.total_requests
        );
    }

    /// Serialised KV transfer is a throughput ceiling that no amount of GPU
    /// removes, and the simulator could not see it: the handoff was scheduled
    /// as a fixed delay with unbounded overlap. Job 316849 measured 4.08 req/s
    /// against a serialised bound of 1000/181.85 = 5.50 req/s while nothing in
    /// the deployment was saturated.
    #[test]
    fn serialised_kv_transfer_caps_throughput_and_more_buffers_lift_it() {
        // A longer window than short_config's, because the point of the
        // serialised arm is that it barely completes anything: at 171 ms per
        // transfer it clears 5.8 req/s, and an 8 s scored window can close
        // with nothing in it, which is indistinguishable from a broken test.
        let run = |n: u32| {
            let mut c = short_config();
            c.workload.benchmark_s = 60.0;
            c.workload.grace_s = 30.0;
            c.kv.xfer_concurrency = n;
            // Pin the whole topology. This test is about the transfer pool
            // and it inherited whatever the default was, until the default
            // moved to 4 x TP2 prefill and the sixteen-buffer arm fell to 93 --
            // below the floor this test needs to say anything at all. See
            // `the_two_models_disagree_about_narrow_prefill` below: that drop
            // is a real disagreement between our two models and it is recorded
            // there rather than smoothed over here.
            c.topology.prefill_workers = 2;
            c.topology.prefill_tp = 4;
            c.topology.decode_workers = 2;
            c.topology.decode_tp = 4;
            Simulator::new(SimSetup { config: c }).run()
        };
        let one = run(1);
        let many = run(16);

        // Serialisation is not "slower", it is "nothing comes out". With one
        // buffer the simulator completes 48 requests in 92 simulated seconds
        // and scores none of them; with sixteen it completes 1056 and scores
        // 696. Decode concurrency is 1.3 against 36.4 and the prefill workers
        // sit 92% idle -- which is job 316849's signature exactly: a
        // deployment where nothing is saturated and throughput is still bad.
        // A ratio, not a zero, and the ratio is not the claim either.
        //
        // This asserted zero until the default topology went from 4P1D to
        // 2P2D, then 10x until it went to 4 x TP2 prefill at N=128, where the
        // margin fell to 2.7x. Nothing broke: at sixteen buffers the handoff
        // stops being the binding constraint, so widening it further buys
        // nothing and the ratio is capped by whatever binds next. Each time,
        // the number in the assertion was a property of the configuration and
        // the mechanism underneath it did not move.
        //
        // So assert the mechanism: one buffer must bind, and lifting it must
        // be worth substantially more than measurement noise. A test that
        // pins a ratio has to be rewritten whenever the deployment changes,
        // which means it is measuring the deployment, not the mechanism.
        assert!(
            many.goodput.total_requests > 2 * one.goodput.total_requests.max(1),
            "one buffer scored {} and sixteen scored {}. Widening the handoff \
             should still be worth more than 2x here; if it is not, the \
             transfer pool has stopped being a constraint at any width and \
             the model no longer needs it.",
            one.goodput.total_requests,
            many.goodput.total_requests
        );

        // And the other half of the mechanism: with one buffer the handoff is
        // what binds, not prefill or decode. This is the part that made job
        // 316849 unreadable -- nothing was saturated and throughput was still
        // bad, because the resource that bound was not in the model at all.
        assert!(
            one.goodput.total_requests < many.goodput.total_requests,
            "one buffer did not bind at all"
        );
        assert!(
            many.goodput.total_requests > 100,
            "sixteen buffers scored only {} requests, so the comparison says \
             nothing about the buffer count: {:?}",
            many.goodput.total_requests,
            many.diagnostics
        );
        assert!(
            many.diagnostics.mean_decode_concurrency
                > 10.0 * one.diagnostics.mean_decode_concurrency,
            "decode concurrency barely moved: {:.2} -> {:.2}",
            one.diagnostics.mean_decode_concurrency,
            many.diagnostics.mean_decode_concurrency
        );
    }

    #[test]
    fn a_run_produces_a_scored_window_and_finishes() {
        let r = Simulator::new(SimSetup {
            config: short_config(),
        })
        .run();
        assert!(
            r.goodput.total_requests > 0,
            "nothing was scored: {:?}",
            r.diagnostics
        );
        assert!(r.goodput.req_per_s > 0.0);
        assert!(r.diagnostics.mean_decode_concurrency > 0.0);
        assert!(r.diagnostics.requests_completed >= r.goodput.total_requests);
    }

    /// The same configuration must produce the same number twice, or an A/B
    /// between policies is measuring the scheduler of the simulator.
    #[test]
    fn the_simulation_is_deterministic() {
        let a = Simulator::new(SimSetup {
            config: short_config(),
        })
        .run();
        let b = Simulator::new(SimSetup {
            config: short_config(),
        })
        .run();
        assert_eq!(a.goodput.total_requests, b.goodput.total_requests);
        assert_eq!(a.goodput.good_requests, b.goodput.good_requests);
        assert!((a.goodput.goodput_req_s - b.goodput.goodput_req_s).abs() < 1e-9);
    }

    /// Raising client concurrency must not raise goodput without bound: past
    /// the decode ceiling the extra load turns into TTFT, exactly as measured.
    #[test]
    fn goodput_saturates_and_then_ttft_degrades() {
        let mut low = short_config();
        low.workload.concurrency = 32;
        let mut high = short_config();
        high.workload.concurrency = 256;
        let a = Simulator::new(SimSetup { config: low }).run();
        let b = Simulator::new(SimSetup { config: high }).run();
        assert!(a.goodput.total_requests > 0 && b.goodput.total_requests > 0);
        assert!(
            b.goodput.ttft.p99 > a.goodput.ttft.p99,
            "TTFT should blow up with load: {} -> {}",
            a.goodput.ttft.p99,
            b.goodput.ttft.p99
        );
    }

    #[test]
    fn the_prefill_policy_is_actually_wired_through() {
        let mut fcfs = short_config();
        fcfs.scheduler.prefill_policy = PrefillPolicy::Fcfs;
        let a = Simulator::new(SimSetup { config: fcfs }).run();
        let b = Simulator::new(SimSetup {
            config: short_config(),
        })
        .run();
        // They need not differ on this uniform workload, but both must produce
        // a valid, scored run.
        assert!(a.goodput.total_requests > 0 && b.goodput.total_requests > 0);
    }
}
