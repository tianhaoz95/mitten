use serde::Deserialize;

/// Text configuration for Gemma 4 (parsed from model config.json text_config).
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub sliding_window: usize,
    pub layer_types: Vec<String>,
    pub eos_token_id: u32,
}

/// Top-level model config wrapping text_config.
#[derive(Debug, Deserialize)]
pub struct Gemma4Config {
    pub text_config: Gemma4TextConfig,
}

impl Gemma4Config {
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}
