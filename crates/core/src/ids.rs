use std::fmt;

use serde::{Deserialize, Serialize};

/// Opaque request identifier. Cheap to copy and to hash.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RequestId(pub u64);

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "req-{}", self.0)
    }
}

impl fmt::Debug for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Identifier of a prefill or decode worker inside one deployment.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WorkerId(pub u32);

impl fmt::Display for WorkerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "w{}", self.0)
    }
}

impl fmt::Debug for WorkerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Monotonic, single-process request id source.
#[derive(Debug, Default)]
pub struct RequestIdSource(std::sync::atomic::AtomicU64);

impl RequestIdSource {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn next(&self) -> RequestId {
        RequestId(self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}
