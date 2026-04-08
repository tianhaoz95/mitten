use crate::backend::{BackendHandle, Logits, KvPool, BoxFuture};
use crate::batch::Batch;
use crate::config::ModelConfig;
use crate::error::EngineError;
use burn::prelude::*;
use burn_wgpu::{Wgpu, WgpuDevice};

pub struct WgpuBackend<B: Backend> {
    config: ModelConfig,
    device: B::Device,
    pool: WgpuKvPool,
}

impl WgpuBackend<Wgpu> {
    pub fn new(config: ModelConfig, device: WgpuDevice, num_pages: usize) -> Self {
        Self { 
            config, 
            device,
            pool: WgpuKvPool { total: num_pages, free: num_pages }
        }
    }
}

impl BackendHandle for WgpuBackend<Wgpu> {
    fn forward(&self, batch: &Batch) -> BoxFuture<'_, Result<Logits, EngineError>> {
        let num_requests = batch.requests.len();
        let vocab_size = self.config.vocab_size;
        let device = self.device.clone();
        
        // Clone the requests to move them into the async block
        let requests = batch.requests.clone();
        
        Box::pin(async move {
            // Simulate GPU work
            let _dummy = Tensor::<Wgpu, 1>::zeros([1024], &device);

            let mut data = vec![0.0f32; num_requests * vocab_size];
            for i in 0..num_requests {
                let req = requests[i].lock();
                let req_id_hash = req.id.as_u128() as u64;
                let step = req.output_ids.len() as u64;
                // Use a simple LCG to vary the token per request and step
                let token = ((req_id_hash.wrapping_mul(6364136223846793005).wrapping_add(step.wrapping_mul(1442695040888963407))) % vocab_size as u64) as usize;
                data[i * vocab_size + token] = 10.0;
            }

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

pub struct WgpuKvPool {
    pub total: usize,
    pub free: usize,
}

impl KvPool for WgpuKvPool {
    fn free_pages(&self) -> usize {
        self.free
    }

    fn total_pages(&self) -> usize {
        self.total
    }
}
