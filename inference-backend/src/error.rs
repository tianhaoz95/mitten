use thiserror::Error;

#[derive(Debug, Error, Clone)]
pub enum EngineError {
    #[error("backend error: {0}")]
    Backend(String),
    #[error("out of KV pages")]
    OutOfKvPages,
    #[error("request aborted: {0:?}")]
    Aborted(AbortReason),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbortReason {
    ClientDisconnected,
    MaxTokensExceeded,
    BackendError(String),
    Preempted,
}
