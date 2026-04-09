use burn::prelude::*;
use crate::backend::{BackendHandle, BoxFuture, KvPool, Logits};
use crate::batch::Batch;
use crate::config::ModelConfig;
use crate::error::EngineError;
use crate::request::RequestState;
use inference_model_common::{InferenceModel, KVCacheState};
use std::sync::Mutex;
use std::collections::HashMap;
use crate::request::RequestId;

pub struct BurnBackend<M: InferenceModel<B>, B: Backend> {
    model: Mutex<M>,
    config: ModelConfig,
    device: B::Device,
    kv_cache: Mutex<HashMap<RequestId, Vec<KVCacheState<B>>>>,
}

impl<M: InferenceModel<B>, B: Backend> BurnBackend<M, B> {
    pub fn new(model: M, config: ModelConfig, device: B::Device) -> Self {
        Self {
            model: Mutex::new(model),
            config,
            device,
            kv_cache: Mutex::new(HashMap::new()),
        }
    }
}

impl<M: InferenceModel<B> + Send + 'static, B: Backend + 'static> BackendHandle for BurnBackend<M, B> {
    fn forward(&self, batch: &Batch) -> BoxFuture<'_, Result<Logits, EngineError>> {
        let num_requests = batch.requests.len();
        let vocab_size = self.config.vocab_size;
        let num_layers = self.config.num_layers;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let device = self.device.clone();

        let mut all_data = Vec::with_capacity(num_requests * vocab_size);
        let mut kv_caches = self.kv_cache.lock().unwrap();
        let model = self.model.lock().unwrap();

        let mut offset = 0;
        for req_arc in &batch.requests {
            let req = req_arc.lock();
            let extend_len = if matches!(req.state, RequestState::Prefilling { .. }) { req.extend_len } else { 1 };

            match &req.state {
                RequestState::Prefilling { processed_tokens } => {
                    let total = req.input_ids.len();
                    eprintln!(">> [prefill] {}/{} tokens", processed_tokens + extend_len, total);
                }
                RequestState::Decoding => {
                    eprintln!(">> [decode] {} tokens generated", req.output_ids.len() + 1);
                }
                _ => {}
            }
            
            let ids_vec: Vec<i32> = batch.input_ids[offset..offset+extend_len].iter().map(|&id| id as i32).collect();
            let pos_vec: Vec<i32> = batch.position_ids[offset..offset+extend_len].iter().map(|&id| id as i32).collect();
            offset += extend_len;

            let ids = Tensor::<B, 2, Int>::from_data(TensorData::new(ids_vec, [1, extend_len]), &device);
            let pos = Tensor::<B, 1, Int>::from_data(TensorData::new(pos_vec, [extend_len]), &device);

            let cache = kv_caches.entry(req.id).or_insert_with(|| {
                model.init_cache(&device)
            });

            let logits_3d = model.forward(ids, pos, cache);
            
            // Get last token logits
            let last_logits = logits_3d.slice([0..1, (extend_len-1)..extend_len]);
            let logits_data = last_logits.into_data();
            all_data.extend(logits_data.iter::<f32>());
        }

        let logits = Logits {
            data: all_data,
            num_rows: num_requests,
            vocab_size,
        };

        Box::pin(async move { Ok(logits) })
    }

    fn kv_pool(&self) -> &dyn KvPool {
        &BurnKvPool { total: 512, free: 512 }
    }

    fn model_config(&self) -> &ModelConfig {
        &self.config
    }
}

struct BurnKvPool { total: usize, free: usize }
impl KvPool for BurnKvPool {
    fn free_pages(&self) -> usize { self.free }
    fn total_pages(&self) -> usize { self.total }
}
