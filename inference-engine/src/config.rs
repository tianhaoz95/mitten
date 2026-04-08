use inference_backend::ModelConfig;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Maximum total tokens (prompt + generated) across all requests in flight.
    pub max_total_tokens: usize,
    /// Number of KV cache pages in the pool.
    pub num_kv_pages: usize,
    /// Tokens per KV cache page.
    pub page_size: usize,
    /// Maximum tokens to process in one prefill iteration (token budget).
    pub max_prefill_tokens_per_iter: usize,
    /// Maximum requests in a single decode batch.
    pub max_decode_batch_size: usize,
    /// Model architecture metadata.
    pub model_config: ModelConfig,
}

#[derive(Debug, Default)]
pub struct EngineStats {
    pub total_requests_served: u64,
    pub total_tokens_generated: u64,
    pub cache_hit_tokens: u64,
    pub cache_miss_tokens: u64,
    pub current_waiting: usize,
    pub current_prefilling: usize,
    pub current_decoding: usize,
}
