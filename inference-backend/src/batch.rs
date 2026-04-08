use crate::request::Request;
use parking_lot::Mutex;
use std::sync::Arc;

/// A batch handed from the scheduler to the engine for a single forward pass.
#[derive(Debug)]
pub struct Batch {
    /// Whether this batch contains prefill tokens, decode tokens, or both
    /// (continuous batching mixes them).
    pub phase: BatchPhase,

    /// Requests participating in this batch, in order.
    /// The engine uses this to interpret `input_ids` and `page_table_entries`.
    pub requests: Vec<Arc<Mutex<Request>>>,

    /// Flat list of token IDs to process.
    /// For prefill requests: their `extend_len` new tokens.
    /// For decode requests: their last generated token (single token each).
    pub input_ids: Vec<u32>,

    /// Position IDs matching `input_ids`. Needed for RoPE.
    pub position_ids: Vec<u32>,

    /// For each request, the KV page indices it owns, used by the attention
    /// kernel to locate KV entries in the pool.
    pub page_table: Vec<Vec<u32>>,

    /// Total number of KV slots consumed by this batch.
    /// Used by the engine to validate the pool has enough space.
    pub num_kv_slots: usize,

    /// For MoE models: pre-computed expert routing decisions (optional).
    /// If Some, the engine uses these directly; if None, routing is computed
    /// during the forward pass.
    pub expert_routing: Option<ExpertRouting>,
}

impl Clone for Batch {
    fn clone(&self) -> Self {
        Self {
            phase: self.phase,
            requests: self.requests.clone(),
            input_ids: self.input_ids.clone(),
            position_ids: self.position_ids.clone(),
            page_table: self.page_table.clone(),
            num_kv_slots: self.num_kv_slots,
            expert_routing: None, // ExpertRouting is not Clone in this stub
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchPhase {
    /// All requests are in the prefill phase.
    PrefillOnly,
    /// All requests are in the decode phase.
    DecodeOnly,
    /// Mix of prefill and decode (continuous batching).
    Mixed,
}

/// Pre-computed expert assignments for a MoE forward pass.
/// Computed by the router during scheduling to allow expert weight prefetching.
#[derive(Debug)]
pub struct ExpertRouting {
    /// For each token, the selected expert indices (top-k per token).
    /// Shape: [num_tokens, top_k]
    pub token_expert_ids: Vec<Vec<u32>>,
    /// Which experts are needed across the whole batch.
    pub unique_expert_ids: Vec<u32>,
}
