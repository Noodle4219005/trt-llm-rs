use crate::ids::{RequestId, WorkerId};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no worker available for role {0}")]
    NoWorker(&'static str),

    #[error("worker {0} is not registered")]
    UnknownWorker(WorkerId),

    #[error("request {0} is not known to this component")]
    UnknownRequest(RequestId),

    #[error("kv cache exhausted: needed {needed} blocks, {free} free")]
    KvExhausted { needed: usize, free: usize },

    #[error("request {id} rejected: {reason}")]
    Rejected { id: RequestId, reason: String },

    #[error("engine backend error: {0}")]
    Engine(String),

    #[error("kv transfer failed: {0}")]
    Transfer(String),

    #[error("configuration error: {0}")]
    Config(String),

    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
