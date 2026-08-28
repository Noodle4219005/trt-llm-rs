//! Worker runtimes.
//!
//! A prefill worker and a decode worker are each one async task around an
//! [`trtllm_engine::Engine`] and the matching scheduler. [`Deployment`] wires a
//! router, several prefill workers and a decode worker together in one process
//! and implements the frontend's [`trtllm_frontend::Completions`] trait, so the
//! whole control plane can be exercised over real HTTP with no GPUs attached.
//!
//! The same tasks run in production; only the `Engine` and the transport
//! between them change. That is deliberate - a deployment topology that only
//! exists in the distributed configuration is a topology nobody can debug.

pub mod decode_worker;
pub mod deployment;
pub mod prefill_worker;
pub mod tokenizer;

pub use deployment::{Deployment, DeploymentHandle};
pub use tokenizer::{SyntheticTokenizer, Tokenizer};
