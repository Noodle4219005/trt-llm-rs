//! OpenAI-compatible HTTP frontend.
//!
//! The frontend does three jobs and no more: parse, stamp the arrival time, and
//! stream. In particular it does *not* queue - queueing is the scheduler's job,
//! and a hidden queue in the HTTP layer is a queue whose waiting time never
//! reaches the deadline arithmetic. The arrival timestamp is taken the moment
//! the request is accepted, which is the same instant the benchmark client
//! starts its own TTFT stopwatch; anything later flatters every number
//! downstream.

pub mod api;
pub mod server;

pub use api::{Completions, GenerateRequest, StreamChunk};
pub use server::{serve, AppState};
