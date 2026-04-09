use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3_5RopeParameters {
    pub rope_theta: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3_5TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub eos_token_id: u32,
    pub rope_parameters: Option<Qwen3_5RopeParameters>,
}

impl Qwen3_5TextConfig {
    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters.as_ref().map(|p| p.rope_theta).unwrap_or(1000000.0)
    }
}

#[derive(Debug, Deserialize)]
pub struct Qwen3_5Config {
    pub text_config: Qwen3_5TextConfig,
}

impl Qwen3_5Config {
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}
