//! Core types for `trt-llm-rs`.
//!
//! Everything in this crate is deliberately free of I/O and of `tokio`, so that
//! the scheduler and the discrete-event simulator can share exactly the same
//! code paths as the production workers.
//!
//! Time is always an explicit `Millis` argument rather than a call to
//! `Instant::now()`. That is what makes a policy testable at zero GPU cost:
//! the simulator advances virtual time, the worker passes wall-clock time, and
//! neither one needs a different scheduler.

pub mod capacity;
pub mod config;
pub mod engine_config;
pub mod error;
pub mod ids;
pub mod remedy;
pub mod request;
pub mod slo;
pub mod stats;
pub mod verdict;

pub use capacity::{CapacityModel, PdSplit};
pub use error::{Error, Result};
pub use ids::{RequestId, WorkerId};
pub use request::{Phase, Request, RequestOutcome, SamplingParams};
pub use slo::{Slo, Verdict};
pub use stats::{GoodputReport, LatencyStats};

/// Milliseconds since an arbitrary epoch. `f64` so virtual and wall clocks are
/// interchangeable and sub-millisecond scheduling decisions stay exact.
pub type Millis = f64;

/// A token id as produced by the tokenizer.
pub type TokenId = u32;
