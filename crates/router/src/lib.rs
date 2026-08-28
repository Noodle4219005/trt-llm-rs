//! Routing.
//!
//! The router picks a prefill worker and a decode worker for each request. It
//! does so in **milliseconds of predicted time to first token**, not in
//! arbitrary load units, because that is the only currency in which queue depth
//! and prefix reuse are comparable:
//!
//! ```text
//! predicted_ttft(worker) = queued_tokens / rate          <- wait for the queue to drain
//!                        + (prompt_len - prefix_hit) / rate   <- our own compute
//!                        + kv_transfer_ms                     <- handoff to decode
//! ```
//!
//! A cache-affinity *bonus* - the usual formulation - needs a weight nobody can
//! derive, and the weight is wrong as soon as the queue depth changes. Counting
//! a prefix hit as "tokens we do not have to compute" needs no weight at all:
//! it is already in milliseconds.
//!
//! With `--cache-bust` the prefix term is identically zero and this degenerates
//! to least-predicted-wait, which is the correct behaviour for the scored run.

pub mod policy;
pub mod registry;

pub use policy::{Router, RouterTuning, RoutingDecision, RoutingError};
pub use registry::{WorkerLoad, WorkerRegistry, WorkerRole, WorkerState};
