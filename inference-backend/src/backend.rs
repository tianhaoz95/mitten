use crate::batch::Batch;
use crate::config::ModelConfig;
use crate::error::EngineError;
use std::future::Future;
use std::pin::Pin;

/// Raw logits output from a forward pass.
pub struct Logits {
    pub data: Vec<f32>,
    pub num_rows: usize,
    pub vocab_size: usize,
}

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

pub trait BackendHandle: Send + Sync + 'static {
    fn forward(&self, batch: &Batch) -> BoxFuture<'_, Result<Logits, EngineError>>;
    fn kv_pool(&self) -> &dyn KvPool;
    fn model_config(&self) -> &ModelConfig;
    fn prefetch(&self, _batch: &Batch) {}
}

pub trait KvPool: Send + Sync {
    fn free_pages(&self) -> usize;
    fn total_pages(&self) -> usize;
}
