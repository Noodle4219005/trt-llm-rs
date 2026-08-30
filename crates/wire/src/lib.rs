//! The worker wire protocol: how the Rust control plane talks to one engine
//! process.
//!
//! **Not to be confused with `trtllm-transfer`**, which moves KV cache *between*
//! workers. This crate is the request path *to* a worker: submit a request,
//! stream tokens back, cancel exactly once.
//!
//! It lives in its own crate rather than inside `trtllm-dynamo` because two
//! unrelated consumers need it and only one of them may depend on Dynamo:
//!
//! * `trtllm-dynamo` wraps it in a Dynamo `LLMEngine` (the Phase D adapter).
//! * `trtllm-worker`'s serving path uses it directly, and must not pull in
//!   `dynamo-backend-common`. That is not a preference: Cargo parses the
//!   manifest of every optional path dependency, and
//!   `third_party/dynamo/lib/backend-common` inherits `authors` from the Dynamo
//!   workspace root, which fails to resolve against ours. An optional feature
//!   cannot avoid that -- only not depending on the crate can.
//!
//! The [`Transport`] trait is deliberately narrow, and [`RequestStream`] owns
//! the lifecycle (terminal event, exactly-once cancel, cleanup on drop) so that
//! no caller can get it subtly wrong.

pub mod telemetry;
pub mod transport;

pub use telemetry::RequestTelemetry;
pub use transport::{
    DynamoAdapter, RequestStream, SseEvent, StreamOutput, Transport, TransportError,
    TransportFactory, TransportRequest,
};

#[cfg(feature = "http-transport")]
pub use transport::HttpTransportFactory;
