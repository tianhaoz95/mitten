#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub is_moe: bool,
    pub num_experts: Option<usize>,
    pub top_k_experts: Option<usize>,
    pub eos_token_ids: Vec<u32>,
}
