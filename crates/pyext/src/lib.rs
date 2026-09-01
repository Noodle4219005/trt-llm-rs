//! `trtllm_rs`: a Python extension module handing TensorRT-LLM's
//! `SimpleScheduler` a Rust-backed capacity decision.
//!
//! This is the opposite direction from `crates/pyhost`, which embeds CPython
//! inside Rust. Here Rust is the thing CPython `dlopen()`s: `PyExecutor`
//! keeps its own step loop and calls into [`RustScheduler::decide`] once per
//! step.
//!
//! ## Hook point: `capacity_scheduler`, not the whole `RequestScheduler`
//!
//! `SimpleScheduler.schedule_request` (scheduler/scheduler.py:406) is two
//! calls:
//! ```text
//! fitting, fitting_disagg_gen_init, paused = self.capacity_scheduler.schedule_request(active_requests)
//! encoder, context, generation = self.micro_batch_scheduler.schedule(fitting, inflight_request_ids)
//! ```
//! Admission policy -- the thing this crate takes over, and where the
//! measured 14% headroom lives -- is entirely inside `capacity_scheduler`.
//! `micro_batch_scheduler` owns chunked-prefill chunk sizing (it assigns
//! `req.context_chunk_size`, scheduler.py:604/682), the context/generation
//! split, and encoder requests -- mechanism this crate does not reproduce.
//! Replacing the whole `RequestScheduler` would mean reproducing all of
//! that, and silently forgetting `context_chunk_size` would stop context
//! requests advancing at all (a hang, not a scheduling bug). So
//! [`RustScheduler::decide`] answers exactly the `capacity_scheduler`
//! question -- admitted or paused -- and is phase-agnostic in its output:
//! `fitting_indices` mixes context and generation indices. It does not
//! produce `fitting_disagg_gen_init`; whether a disaggregated gen-init
//! request fits is a KV-capacity/transfer-readiness question about
//! mechanism, and stays on the Python side, delegated to the upstream
//! `capacity_scheduler`.
//!
//! ## Why the boundary is crossed exactly once per step
//!
//! `crates/pyhost`'s own tests measured a PyO3 round trip at 9,533 ns. At 60
//! requests x 5 attributes, reading each attribute off each Python request
//! object individually would be 300 crossings per step -- about 2.9 ms of
//! pure boundary overhead before any scheduling work happens. So the contract
//! is: Python flattens its request objects into plain arrays with cheap
//! C-level list comprehensions, hands them to [`RustScheduler::decide`] in
//! one call, and gets back index lists it uses to re-index its own request
//! list. No per-request Python attribute access happens on the Rust side.
//!
//! ## What `decide` actually does
//!
//! The scheduling policy itself already exists in `crates/sched` and is not
//! reimplemented here:
//! * Generation admission is governed by the measured decode headroom in
//!   [`trtllm_sched::ItlController`] (`crates/sched/src/decode.rs:3-7`): 53
//!   sequences at a mean ITL of 17.23 ms leave 14% of a 20 ms budget unspent,
//!   and the controller's AIMD cap is where that headroom gets spent.
//! * Everything else ([`decide::decide_indices`]) is bookkeeping: partition
//!   candidates, respect `max_batch_size` and the shared `max_num_tokens`
//!   budget (a context request charges `prompt_len - context_done_tokens`,
//!   a generation request charges 1), and do it in a deterministic order so
//!   that every TP rank, running this same computation independently,
//!   agrees.

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use trtllm_core::Millis;
use trtllm_sched::ItlController;

mod decide;

use decide::{decide_indices, validate_lengths, DecideInput, DecodeOrder};

/// A Rust-backed capacity decision for TensorRT-LLM's `SimpleScheduler`,
/// standing in for its `capacity_scheduler` (not the whole `RequestScheduler`
/// -- see the crate docs for why).
///
/// All state a scheduling decision needs across steps -- the ITL controller
/// and the running admit/pause counters -- lives here, inside the
/// `#[pyclass]`, exactly as the Python side expects of the object it holds.
#[pyclass]
pub struct RustScheduler {
    max_batch_size: usize,
    max_num_tokens: usize,
    kv_total_blocks: usize,
    /// The scored mean-ITL budget. The controller steers on it and the slack
    /// ordering ranks on it, so it must be the same number in both places.
    itl_budget_ms: f64,
    /// Which running request gives up its step when the cap binds. Read from
    /// TRTLLM_RS_DECODE_ORDER so it can be A/B'd without a rebuild -- the
    /// worker binary is staged into a container and a rebuild is the
    /// expensive part of a comparison.
    decode_order: DecodeOrder,
    /// Free blocks below which a request that has not started is held back.
    /// `kv_free_blocks` crossed the ABI from the first version and was stored
    /// and never read.
    kv_watermark_blocks: usize,
    controller: ItlController,
    admitted_total: u64,
    paused_total: u64,
    steps_observed: u64,
    last_kv_free_blocks: usize,
    last_now_ms: Millis,
}

#[pymethods]
impl RustScheduler {
    #[new]
    pub fn new(
        max_batch_size: usize,
        max_num_tokens: usize,
        itl_budget_ms: f64,
        kv_total_blocks: usize,
    ) -> Self {
        // Start optimistic (cap == max_batch_size) and let the AIMD
        // controller back off once it has enough step-latency samples
        // (ItlController holds at the initial cap for its first 8
        // observations -- see crates/sched/src/decode.rs). `max_batch_size`
        // is enforced independently by `decide`, so an initial cap this high
        // never itself admits more than the batch allows.
        let max_cap = (max_batch_size.max(1)) as f64;
        let decode_order = match std::env::var("TRTLLM_RS_DECODE_ORDER").as_deref() {
            Ok("arrival") => DecodeOrder::Arrival,
            // Anything else, including unset and a typo, gets the default.
            // A misspelt policy that silently disabled the scheduler would be
            // the worst of both: no benefit and no signal.
            _ => DecodeOrder::Slack,
        };
        RustScheduler {
            max_batch_size,
            max_num_tokens,
            kv_total_blocks,
            itl_budget_ms,
            decode_order,
            // 5% of the pool, matching KvConfig::admission_watermark, which
            // the simulator and the worker have both used since the beginning
            // while the live path did not.
            kv_watermark_blocks: kv_total_blocks / 20,
            controller: ItlController::new(itl_budget_ms, max_cap, 1.0, max_cap),
            admitted_total: 0,
            paused_total: 0,
            steps_observed: 0,
            last_kv_free_blocks: kv_total_blocks,
            last_now_ms: 0.0,
        }
    }

    /// One capacity decision for one `capacity_scheduler.schedule_request`
    /// call (see the crate docs for why this is the hook point, not the
    /// whole `RequestScheduler`). Every argument is a flat array rather than
    /// a list of request objects; see the crate docs for why.
    #[allow(clippy::too_many_arguments)]
    pub fn decide(
        &mut self,
        ids: Vec<i64>,
        is_context: Vec<bool>,
        prompt_lens: Vec<u32>,
        context_done_tokens: Vec<u32>,
        tokens_generated: Vec<u32>,
        max_new_tokens: Vec<u32>,
        arrival_ms: Vec<f64>,
        inflight: Vec<i64>,
        kv_free_blocks: usize,
        now_ms: f64,
    ) -> PyResult<(Vec<usize>, Vec<usize>)> {
        let n = ids.len();
        validate_lengths(
            n,
            &[
                ("is_context", is_context.len()),
                ("prompt_lens", prompt_lens.len()),
                ("context_done_tokens", context_done_tokens.len()),
                ("tokens_generated", tokens_generated.len()),
                ("max_new_tokens", max_new_tokens.len()),
                ("arrival_ms", arrival_ms.len()),
            ],
        )
        .map_err(PyValueError::new_err)?;

        // These three were length-checked and discarded as "reserved" while
        // the ordering was arrival-then-id. They are what the slack ordering
        // needs, and the bridge has been computing them the whole time.
        let generation_cap = self.controller.cap().max(0.0) as usize;
        let input = DecideInput {
            ids: &ids,
            is_context: &is_context,
            prompt_lens: &prompt_lens,
            context_done_tokens: &context_done_tokens,
            arrival_ms: &arrival_ms,
            inflight: &inflight,
            max_batch_size: self.max_batch_size,
            max_num_tokens: self.max_num_tokens,
            generation_cap,
            tokens_generated: &tokens_generated,
            max_new_tokens: &max_new_tokens,
            now_ms,
            itl_budget_ms: self.itl_budget_ms,
            decode_order: self.decode_order,
            kv_free_blocks,
            kv_watermark_blocks: self.kv_watermark_blocks,
        };
        let (fitting_indices, paused_indices) = decide_indices(&input);

        self.admitted_total += fitting_indices.len() as u64;
        self.paused_total += paused_indices.len() as u64;
        self.last_kv_free_blocks = kv_free_blocks;
        self.last_now_ms = now_ms;

        Ok((fitting_indices, paused_indices))
    }

    /// Feed one measured step time back to the `ItlController`. `concurrency`
    /// is how many sequences were actually in that step's batch -- see
    /// `ItlController::observe`'s docs for why that matters (a cap that is
    /// not the binding constraint must not grow).
    pub fn observe(&mut self, step_ms: f64, concurrency: usize) {
        self.controller.observe(step_ms, concurrency);
        self.steps_observed += 1;
    }

    pub fn stats(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("admitted".to_string(), self.admitted_total as f64);
        m.insert("refused".to_string(), self.paused_total as f64);
        m.insert("cap".to_string(), self.controller.cap());
        // Reserved for parity with DecodeScheduler::finish_disagreements
        // (crates/sched/src/decode.rs). This scheduler is handed a full
        // per-request snapshot every call rather than incremental
        // advanced/finished events, so there is no bookkeeping for the
        // engine's own finish reports to disagree with; always 0 here.
        m.insert("finish_disagreements".to_string(), 0.0);
        m.insert(
            "observed_itl_ms".to_string(),
            self.controller.observed_itl_ms(),
        );
        m.insert("samples".to_string(), self.controller.samples() as f64);
        // Observable so a run cannot be read as "the policy chose this cap".
        // 1.0 means the controller stopped steering because concurrency was
        // shown not to move ITL.
        m.insert(
            "concurrency_not_binding".to_string(),
            if self.controller.concurrency_not_binding() {
                1.0
            } else {
                0.0
            },
        );
        m.insert("steps_observed".to_string(), self.steps_observed as f64);
        m.insert("kv_total_blocks".to_string(), self.kv_total_blocks as f64);
        m.insert(
            "kv_free_blocks_last".to_string(),
            self.last_kv_free_blocks as f64,
        );
        m.insert("last_now_ms".to_string(), self.last_now_ms);
        m
    }
}

/// The Python module. Its name must be exactly `trtllm_rs` -- PyO3 derives
/// the `PyInit_trtllm_rs` symbol from this function's name, and that symbol
/// is what `import trtllm_rs` looks for once the built artifact is placed on
/// `sys.path` as `trtllm_rs.so` (see scripts/build-pyext.sbatch).
#[pymodule]
fn trtllm_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustScheduler>()?;
    Ok(())
}
