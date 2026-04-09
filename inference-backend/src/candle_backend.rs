/// Mitten Inference Backend using Candle.
/// Supports modular model architectures.
use std::path::Path;
use std::sync::Mutex;
use std::collections::HashMap;

use candle_core::{DType, Device, Result as CResult, Tensor};
use candle_nn::VarBuilder;

use crate::backend::{BackendHandle, BoxFuture, KvPool, Logits};
use crate::batch::Batch;
use crate::config::ModelConfig;
use crate::error::EngineError;
use crate::request::{RequestState, RequestId};
use crate::models::common::CacheState;
use crate::models::qwen3_5::Qwen3_5Model;
use crate::models::gemma4::Gemma4TextModel;

enum Model {
    Gemma4(Gemma4TextModel),
    Qwen3_5(Qwen3_5Model),
}

pub struct CandleBackend {
    model: Mutex<Model>,
    config: ModelConfig,
    kv_cache: Mutex<HashMap<RequestId, Vec<CacheState>>>,
}

impl CandleBackend {
    pub fn load(model_dir: &Path) -> Result<Self, EngineError> {
        let config_str = std::fs::read_to_string(model_dir.join("config.json")).map_err(|e| EngineError::Backend(e.to_string()))?;
        let cfg: serde_json::Value = serde_json::from_str(&config_str).map_err(|e| EngineError::Backend(e.to_string()))?;
        
        let device = Device::Cpu;
        let weights = model_dir.join("model.safetensors");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, &device).map_err(|e| EngineError::Backend(format!("load weights: {e}")))? };

        let model_type = cfg["model_type"].as_str().unwrap_or("gemma2");
        let (model, model_config) = if model_type == "qwen3_5" {
            let m = Qwen3_5Model::load(vb, &cfg).map_err(|e| EngineError::Backend(format!("build qwen: {e}")))?;
            let text = &cfg["text_config"];
            let c = ModelConfig {
                num_layers: text["num_hidden_layers"].as_u64().unwrap_or(24) as usize,
                num_kv_heads: 2, head_dim: 256,
                vocab_size: text["vocab_size"].as_u64().unwrap_or(248320) as usize,
                is_moe: false, num_experts: None, top_k_experts: None,
                eos_token_id: text["eos_token_id"].as_u64().unwrap_or(248044) as u32,
            };
            (Model::Qwen3_5(m), c)
        } else {
            let m = Gemma4TextModel::load(vb, &cfg).map_err(|e| EngineError::Backend(format!("build gemma: {e}")))?;
            let text = &cfg["text_config"];
            let c = ModelConfig {
                num_layers: text["num_hidden_layers"].as_u64().unwrap_or(35) as usize,
                num_kv_heads: 1, head_dim: 256,
                vocab_size: text["vocab_size"].as_u64().unwrap_or(262144) as usize,
                is_moe: false, num_experts: None, top_k_experts: None,
                eos_token_id: text["eos_token_id"].as_u64().unwrap_or(1) as u32,
            };
            (Model::Gemma4(m), c)
        };

        Ok(Self { model: Mutex::new(model), config: model_config, kv_cache: Mutex::new(HashMap::new()) })
    }
}

impl BackendHandle for CandleBackend {
    fn forward(&self, batch: &Batch) -> BoxFuture<'_, Result<Logits, EngineError>> {
        let num_requests = batch.requests.len();
        let vocab_size = self.config.vocab_size;
        let num_layers = self.config.num_layers;

        let result = (|| -> Result<Logits, EngineError> {
            let device = Device::Cpu;
            let mut all_data = Vec::with_capacity(num_requests * vocab_size);
            let model = self.model.lock().unwrap();
            let mut kv_caches = self.kv_cache.lock().unwrap();

            let mut offset = 0;
            for req_arc in &batch.requests {
                let req = req_arc.lock();
                let extend_len = if matches!(req.state, RequestState::Prefilling { .. }) { req.extend_len } else { 1 };
                
                let ids_vec = batch.input_ids[offset..offset+extend_len].to_vec();
                let pos_vec = batch.position_ids[offset..offset+extend_len].to_vec();
                offset += extend_len;

                let ids = Tensor::from_vec(ids_vec, (1, extend_len), &device).map_err(|e| EngineError::Backend(e.to_string()))?;
                let pos = Tensor::from_vec(pos_vec, (1, extend_len), &device).map_err(|e| EngineError::Backend(e.to_string()))?;

                match *model {
                    Model::Gemma4(ref m) => {
                        let cache = kv_caches.entry(req.id).or_insert_with(|| {
                            (0..num_layers).map(|_| CacheState::Attention(
                                Tensor::zeros((1, 1, 0, 256), DType::F32, &device).unwrap(),
                                Tensor::zeros((1, 1, 0, 256), DType::F32, &device).unwrap(),
                            )).collect()
                        });
                        let logits_3d = m.forward(&ids, &pos, cache).map_err(|e| EngineError::Backend(e.to_string()))?;
                        let last_logits = logits_3d.narrow(1, extend_len - 1, 1).map_err(|e| EngineError::Backend(e.to_string()))?
                            .flatten_all().map_err(|e| EngineError::Backend(e.to_string()))?
                            .to_vec1::<f32>().map_err(|e| EngineError::Backend(e.to_string()))?;
                        all_data.extend_from_slice(&last_logits);
                    }
                    Model::Qwen3_5(ref m) => {
                        let cache = kv_caches.entry(req.id).or_insert_with(|| {
                            (0..num_layers).map(|i| {
                                if (i + 1) % 4 == 0 {
                                    CacheState::Attention(
                                        Tensor::zeros((1, 2, 0, 256), DType::F32, &device).unwrap(),
                                        Tensor::zeros((1, 2, 0, 256), DType::F32, &device).unwrap(),
                                    )
                                } else {
                                    CacheState::DeltaNet(
                                        Tensor::zeros((1, 16, 128, 128), DType::F32, &device).unwrap(),
                                        Tensor::zeros((1, 6144, 3), DType::F32, &device).unwrap()
                                    )
                                }
                            }).collect()
                        });
                        let logits_3d = m.forward(&ids, &pos, cache).map_err(|e| EngineError::Backend(e.to_string()))?;
                        let last_logits = logits_3d.narrow(1, extend_len - 1, 1).map_err(|e| EngineError::Backend(e.to_string()))?
                            .flatten_all().map_err(|e| EngineError::Backend(e.to_string()))?
                            .to_vec1::<f32>().map_err(|e| EngineError::Backend(e.to_string()))?;
                        all_data.extend_from_slice(&last_logits);
                    }
                }
            }
            Ok(Logits { data: all_data, num_rows: num_requests, vocab_size })
        })();
        Box::pin(async move { result })
    }
    fn kv_pool(&self) -> &dyn KvPool { &CandleKvPool { total: 512, free: 512 } }
    fn model_config(&self) -> &ModelConfig { &self.config }
}

struct CandleKvPool { total: usize, free: usize }
impl KvPool for CandleKvPool {
    fn free_pages(&self) -> usize { self.free }
    fn total_pages(&self) -> usize { self.total }
}
