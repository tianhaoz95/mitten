use burn::prelude::*;
use burn::nn::{Linear, LinearConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig};
use crate::config::Gemma4TextConfig;

#[derive(Module, Debug)]
pub struct Gemma4Model<B: Backend> {
    pub embedding: Embedding<B>,
    pub layers: Vec<Gemma4Layer<B>>,
    pub norm: LayerNorm<B>,
    pub output: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Gemma4Layer<B: Backend> {
    pub attention: Gemma4Attention<B>,
    pub mlp: Gemma4Mlp<B>,
    pub attention_norm: LayerNorm<B>,
    pub mlp_norm: LayerNorm<B>,
}

#[derive(Module, Debug)]
pub struct Gemma4Attention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

#[derive(Module, Debug)]
pub struct Gemma4Mlp<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> Gemma4Model<B> {
    pub fn new(config: &Gemma4TextConfig, device: &B::Device) -> Self {
        let hidden = config.hidden_size;
        let vocab = config.vocab_size;
        let intermediate = config.intermediate_size;
        let head_dim = config.head_dim;
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let eps = config.rms_norm_eps as f32;

        let embedding = EmbeddingConfig::new(vocab, hidden).init(device);
        let norm = LayerNormConfig::new(hidden).with_epsilon(eps as f64).init(device);
        // output shares weights with embedding (tie_word_embeddings=true); use separate Linear for now
        let output = LinearConfig::new(hidden, vocab).with_bias(false).init(device);

        let layers = (0..config.num_hidden_layers).map(|_| {
            Gemma4Layer {
                attention: Gemma4Attention {
                    q_proj: LinearConfig::new(hidden, n_heads * head_dim).with_bias(false).init(device),
                    k_proj: LinearConfig::new(hidden, n_kv_heads * head_dim).with_bias(false).init(device),
                    v_proj: LinearConfig::new(hidden, n_kv_heads * head_dim).with_bias(false).init(device),
                    o_proj: LinearConfig::new(n_heads * head_dim, hidden).with_bias(false).init(device),
                    n_heads,
                    n_kv_heads,
                    head_dim,
                },
                mlp: Gemma4Mlp {
                    gate_proj: LinearConfig::new(hidden, intermediate).with_bias(false).init(device),
                    up_proj: LinearConfig::new(hidden, intermediate).with_bias(false).init(device),
                    down_proj: LinearConfig::new(intermediate, hidden).with_bias(false).init(device),
                },
                attention_norm: LayerNormConfig::new(hidden).with_epsilon(eps as f64).init(device),
                mlp_norm: LayerNormConfig::new(hidden).with_epsilon(eps as f64).init(device),
            }
        }).collect();

        Self { embedding, layers, norm, output }
    }

    /// Forward pass skeleton — full paged attention implementation deferred to Phase 1.
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_batch, _seq] = input_ids.dims();
        let hidden = self.embedding.forward(input_ids);
        // Simplified: skip attention/MLP, just apply final norm + output projection
        let normed = self.norm.forward(hidden);
        let logits = self.output.forward(normed);
        logits
    }
}
