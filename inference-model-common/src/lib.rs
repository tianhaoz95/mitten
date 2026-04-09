use burn::prelude::*;

/// Standardized KV cache states for different model architectures.
#[derive(Debug, Clone)]
pub enum KVCacheState<B: Backend> {
    /// Standard attention cache: (key_cache, value_cache)
    /// Each tensor has shape [batch, heads, seq_len, head_dim]
    Attention(Tensor<B, 4>, Tensor<B, 4>),
    
    /// Specialized cache for DeltaNet or other linear RNN-style architectures
    /// e.g. (state s, conv_cache)
    DeltaNet(Tensor<B, 4>, Tensor<B, 3>),
}

/// The common interface for all Mitten inference models implemented in Burn.
pub trait InferenceModel<B: Backend>: Module<B> {
    type Config: Send + Sync;

    /// Initialize the model from its specific configuration.
    fn new(config: &Self::Config, device: &B::Device) -> Self;

    /// Execute the forward pass with standardized inputs.
    /// Returns logits with shape [batch, seq_len, vocab_size].
    fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        position_ids: Tensor<B, 1, Int>,
        kv_cache: &mut [KVCacheState<B>],
    ) -> Tensor<B, 3>;

    /// Initialize an empty KV cache for this model.
    fn init_cache(&self, device: &B::Device) -> Vec<KVCacheState<B>>;
}

/// Error type for model configuration and loading.
#[derive(Debug)]
pub enum ModelError {
    Config(String),
    Load(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::Config(m) => write!(f, "config error: {}", m),
            ModelError::Load(m) => write!(f, "load error: {}", m),
        }
    }
}

impl std::error::Error for ModelError {}
