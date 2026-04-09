use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3_5RopeParameters {
    pub rope_theta: f64,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
}

fn default_partial_rotary_factor() -> f64 { 1.0 }

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
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub linear_num_key_heads: usize,
    #[serde(default)]
    pub linear_num_value_heads: usize,
    #[serde(default)]
    pub linear_key_head_dim: usize,
    #[serde(default)]
    pub linear_value_head_dim: usize,
}

impl Qwen3_5TextConfig {
    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters.as_ref().map(|p| p.rope_theta).unwrap_or(1000000.0)
    }
    pub fn rotary_dim(&self) -> usize {
        let factor = self.rope_parameters.as_ref().map(|p| p.partial_rotary_factor).unwrap_or(1.0);
        ((self.head_dim as f64) * factor) as usize
    }
    pub fn layer_type(&self, i: usize) -> &str {
        if let Some(types) = &self.layer_types {
            types.get(i).map(|s| s.as_str()).unwrap_or("linear_attention")
        } else if (i + 1) % 4 == 0 { "full_attention" } else { "linear_attention" }
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
