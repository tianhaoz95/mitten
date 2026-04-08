use crate::error::AbortReason;
use tokio::sync::mpsc;
use std::fmt;

pub type RequestId = uuid::Uuid;
pub type PageIndex = u32;
pub type NodeId = usize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestState {
    Waiting,
    Prefilling { processed_tokens: usize },
    Decoding,
    Done,
    Aborted { reason: AbortReason },
}

#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub max_new_tokens: usize,
    pub stop_sequences: Vec<String>,
    pub stream: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            max_new_tokens: 512,
            stop_sequences: vec![],
            stream: false,
        }
    }
}

pub struct Request {
    pub id: RequestId,
    pub input_ids: Vec<u32>,
    pub state: RequestState,

    /// Tokens already in KV cache from radix prefix match. Skip during prefill.
    pub cached_len: usize,
    /// Total tokens with KV entries on device (cached + computed this session).
    pub device_len: usize,
    /// Tokens to compute in the current scheduler iteration (set by PrefillAdder).
    pub extend_len: usize,

    /// KV cache page indices owned by this request.
    pub kv_pages: Vec<PageIndex>,

    /// Tokens generated so far in the decode phase.
    pub output_ids: Vec<u32>,

    pub params: SamplingParams,

    /// Channel to stream token events back to the HTTP handler.
    /// None if streaming is disabled (batch mode).
    pub token_tx: Option<mpsc::Sender<TokenEvent>>,

    /// Radix cache node IDs locked by this request (must call unlock on completion).
    pub locked_nodes: Vec<NodeId>,

    pub arrival_time: std::time::Instant,
    pub first_token_time: Option<std::time::Instant>,
}

impl fmt::Debug for Request {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Request")
            .field("id", &self.id)
            .field("state", &self.state)
            .field("device_len", &self.device_len)
            .field("kv_pages", &self.kv_pages)
            .field("output_ids", &self.output_ids)
            .finish()
    }
}

impl Request {
    pub fn new(id: RequestId, input_ids: Vec<u32>, params: SamplingParams) -> Self {
        Self {
            id,
            input_ids,
            state: RequestState::Waiting,
            cached_len: 0,
            device_len: 0,
            extend_len: 0,
            kv_pages: Vec::new(),
            output_ids: Vec::new(),
            params,
            token_tx: None,
            locked_nodes: Vec::new(),
            arrival_time: std::time::Instant::now(),
            first_token_time: None,
        }
    }
}

#[derive(Debug)]
pub enum TokenEvent {
    Token(u32),
    Done { finish_reason: FinishReason },
    Error(String),
}

#[derive(Debug, Clone)]
pub enum FinishReason {
    EosToken,
    StopString,
    MaxTokens,
    Aborted,
}
