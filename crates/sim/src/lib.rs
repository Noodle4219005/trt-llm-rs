//! A deterministic discrete-event simulation of the whole deployment.
//!
//! This is the cheapest instrument in the project. It runs the *real*
//! schedulers, the real router and the real admission rules against a cost
//! model fitted to measured hardware, in virtual time, in a few milliseconds.
//! A policy change that does not survive here does not get a GPU-hour.
//!
//! What it models faithfully:
//!
//! * closed-loop clients - `N` in flight, each blocking until its request
//!   finishes, which is the only load shape the scored benchmark allows;
//! * per-worker prefill queueing, chunking and batch assembly, including the
//!   deadline-feasibility rule that shrinks batches under pressure;
//! * decode as a stepping batch whose step latency grows with concurrency, with
//!   admission gated by the same [`trtllm_sched::DecodeScheduler`] the workers
//!   run;
//! * KV handoff landing in the *inter-token* budget rather than in TTFT,
//!   because the prefill worker emits the first token itself.
//!
//! What it does **not** model, and must not be asked about: kernel-level effects,
//! memory fragmentation, NCCL behaviour under contention, failure and retry,
//! and anything about the ITL-versus-concurrency curve that was not measured -
//! see [`trtllm_engine::cost::DecodeCurve`] for why that last one is a warning
//! and not a caveat.
//!
//! It also has **no source of variance at all**: fixed ISL and OSL, a
//! deterministic cost per batch, no stragglers, no cold start, no preemption,
//! no fabric jitter. So it reports `good_frac` near 1.0 where real runs of this
//! workload land around 0.93. **Absolute `good_frac` from here is not
//! comparable to a measured run.** What is comparable is the *difference*
//! between two policies under the same conditions, which is what this tool
//! exists for.

mod events;
mod report;
mod sim;

pub use report::{Diagnostics, SimReport};
pub use sim::{SimSetup, Simulator};
