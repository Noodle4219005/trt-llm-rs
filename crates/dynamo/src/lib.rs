//! Dynamo v1.4.1 integration for the Rust control plane.
//!
//! The wire protocol itself is `trtllm-wire`; this crate is only the Dynamo
//! `LLMEngine` adapter around it, plus the config that adapter needs. The split
//! exists so `trtllm-worker` can speak the same protocol without depending on
//! the pinned Dynamo tree.

// Re-exported so every existing path (`trtllm_dynamo::TransportRequest`, the
// integration tests, engine.rs's `use crate::{...}`) keeps working unchanged
// after the move.
pub use trtllm_wire::{telemetry, transport};
pub use trtllm_wire::{
    DynamoAdapter, RequestStream, RequestTelemetry, SseEvent, StreamOutput, Transport,
    TransportError, TransportFactory, TransportRequest,
};

#[cfg(feature = "http-transport")]
pub use trtllm_wire::HttpTransportFactory;

#[cfg(feature = "dynamo-v1")]
pub mod config;
#[cfg(feature = "dynamo-v1")]
pub mod engine;

#[cfg(feature = "dynamo-v1")]
pub use config::DynamoEngineConfig;
#[cfg(feature = "dynamo-v1")]
pub use engine::{run_with_factory, worker_with_factory, DynamoLlmEngine};
