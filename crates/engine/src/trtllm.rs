//! TensorRT-LLM backend, behind the `trtllm` feature.
//!
//! **This module is not built or tested in this tree.** There is no CUDA
//! toolchain and no TensorRT-LLM install here, and a binding that has never
//! been compiled must not be presented as if it worked. What follows is the
//! shape the seam takes, written down so the work is a build problem rather
//! than a design problem. See `docs/trtllm-ffi.md`.
//!
//! The integration point is `tensorrt_llm::executor::Executor` - the C++ API
//! underneath the Python `tensorrt_llm.LLM` class - reached through a narrow
//! `extern "C"` shim compiled by `build.rs`. Narrow on purpose: every type that
//! crosses the boundary is a type that has to be kept in sync with a moving
//! upstream.

use std::ffi::c_void;

use async_trait::async_trait;
use trtllm_core::{Error, Millis, RequestId, Result};

use crate::{DecodeSeqSpec, DecodeStepOutcome, Engine, EngineInfo, PrefillOutcome, PrefillWork};

#[repr(C)]
pub struct TrtllmExecutorHandle {
    _opaque: [u8; 0],
    _marker: std::marker::PhantomData<(*mut c_void, std::marker::PhantomPinned)>,
}

extern "C" {
    /// Returns null on failure and writes a message into `err`.
    fn trtllm_executor_create(
        engine_dir: *const std::ffi::c_char,
        json_config: *const std::ffi::c_char,
        err: *mut std::ffi::c_char,
        err_len: usize,
    ) -> *mut TrtllmExecutorHandle;

    fn trtllm_executor_destroy(handle: *mut TrtllmExecutorHandle);
}

/// Safety contract the shim must uphold for this to be sound:
///
/// * the handle is owned solely by this struct and destroyed exactly once;
/// * the underlying `Executor` is internally synchronised (it is - it owns its
///   own worker threads), so `&self` methods may be called concurrently;
/// * no pointer handed across the boundary outlives the call.
pub struct TrtllmEngine {
    handle: *mut TrtllmExecutorHandle,
    info: EngineInfo,
}

// Sound only under the contract above. Stated explicitly because a wrong
// `unsafe impl Send` here is a data race that shows up as garbage logits hours
// into a run, with no error.
unsafe impl Send for TrtllmEngine {}
unsafe impl Sync for TrtllmEngine {}

impl Drop for TrtllmEngine {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { trtllm_executor_destroy(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

impl TrtllmEngine {
    pub fn open(_engine_dir: &str, _json_config: &str, _info: EngineInfo) -> Result<Self> {
        Err(Error::Engine(
            "the TensorRT-LLM binding is declared but not implemented in this tree; \
             see docs/trtllm-ffi.md"
                .into(),
        ))
    }
}

#[async_trait]
impl Engine for TrtllmEngine {
    fn info(&self) -> EngineInfo {
        self.info.clone()
    }

    async fn prefill(&self, _work: PrefillWork, _now: Millis) -> Result<PrefillOutcome> {
        // The elapsed time returned here must be *measured*, not estimated: the
        // prefill scheduler feeds it straight into its rate EWMA, which feeds
        // every deadline decision downstream.
        Err(Error::Engine("trtllm prefill not implemented".into()))
    }

    async fn add_decode_seq(&self, _seq: DecodeSeqSpec) -> Result<()> {
        Err(Error::Engine(
            "trtllm add_decode_seq not implemented".into(),
        ))
    }

    async fn decode_step(&self, _now: Millis) -> Result<DecodeStepOutcome> {
        Err(Error::Engine("trtllm decode_step not implemented".into()))
    }

    async fn remove_seq(&self, _id: RequestId) -> Result<()> {
        Err(Error::Engine("trtllm remove_seq not implemented".into()))
    }

    async fn decode_concurrency(&self) -> usize {
        0
    }
}
