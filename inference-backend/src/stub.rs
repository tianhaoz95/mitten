use crate::backend::{BackendHandle, Logits, KvPool, BoxFuture};
use crate::batch::Batch;
use crate::config::ModelConfig;
use crate::error::EngineError;

pub struct StubBackend {
    config: ModelConfig,
    pool: StubKvPool,
    rng_seed: u64,
    forced_tokens: Vec<u32>,
}

impl StubBackend {
    pub fn new(config: ModelConfig, num_pages: usize, rng_seed: u64) -> Self {
        Self {
            config,
            pool: StubKvPool {
                total: num_pages,
                free: num_pages,
            },
            rng_seed,
            forced_tokens: Vec::new(),
        }
    }

    pub fn with_forced_tokens(mut self, tokens: Vec<u32>) -> Self {
        self.forced_tokens = tokens;
        self
    }
}

impl BackendHandle for StubBackend {
    fn forward(&self, batch: &Batch) -> BoxFuture<'_, Result<Logits, EngineError>> {
        let num_requests = batch.requests.len();
        let vocab_size = self.config.vocab_size;
        let mut data = vec![0.0f32; num_requests * vocab_size];

        for i in 0..num_requests {
            let req = batch.requests[i].lock();
            let token = if !self.forced_tokens.is_empty() {
                // Return forced tokens in sequence based on current output length
                let idx = req.output_ids.len();
                if idx < self.forced_tokens.len() {
                    self.forced_tokens[idx]
                } else {
                    // Default to EOS
                    self.config.eos_token_id
                }
            } else {
                let req_id_hash = req.id.as_u128() as u64;
                let step = req.output_ids.len() as u64;
                ((self.rng_seed ^ req_id_hash ^ step) % (vocab_size as u64)) as u32
            };
            
            // Ensure token is within vocab range
            let token = (token as usize % vocab_size) as u32;
            data[i * vocab_size + token as usize] = 10.0;
        }

        Box::pin(async move {
            Ok(Logits {
                data,
                num_rows: num_requests,
                vocab_size,
            })
        })
    }

    fn kv_pool(&self) -> &dyn KvPool {
        &self.pool
    }

    fn model_config(&self) -> &ModelConfig {
        &self.config
    }
}

pub struct StubKvPool {
    pub total: usize,
    pub free: usize,
}

impl KvPool for StubKvPool {
    fn free_pages(&self) -> usize {
        self.free
    }

    fn total_pages(&self) -> usize {
        self.total
    }
}
