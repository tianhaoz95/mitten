/// Gemma 4 text-only inference using candle primitives.
/// Weight layout matches model/gemma-4-E2B-it/model.safetensors exactly.
use std::path::Path;
use std::sync::Mutex;

use candle_core::{DType, Device, Module, Result as CResult, Tensor, D};
use candle_nn::{linear_no_bias as linear, Linear, VarBuilder};

use crate::backend::{BackendHandle, BoxFuture, KvPool, Logits};
use crate::batch::Batch;
use crate::config::ModelConfig;
use crate::error::EngineError;

// ── RMSNorm ──────────────────────────────────────────────────────────────────

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> CResult<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }
    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let norm = (x_f32.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?.sqrt()?;
        let x_normed = x_f32.broadcast_div(&norm)?;
        let res = x_normed.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?;
        res.to_dtype(x.dtype())
    }
}

// ── Attention ─────────────────────────────────────────────────────────────────

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    is_global: bool,
    sliding_window: usize,
    rope_theta: f64,
}

impl Attention {
    fn new(vb: VarBuilder, hidden: usize, n_heads: usize, n_kv_heads: usize,
           head_dim: usize, is_global: bool, sliding_window: usize, rope_theta: f64,
           _is_kv_shared: bool) -> CResult<Self> {
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let rotary_dim = if is_global { (head_dim as f32 * 0.25) as usize } else { head_dim };
        Ok(Self {
            q_proj: linear(hidden, q_dim, vb.pp("self_attn.q_proj"))?,
            k_proj: linear(hidden, kv_dim, vb.pp("self_attn.k_proj"))?,
            v_proj: linear(hidden, kv_dim, vb.pp("self_attn.v_proj"))?,
            o_proj: linear(q_dim, hidden, vb.pp("self_attn.o_proj"))?,
            q_norm: RmsNorm::new(head_dim, 1e-6, vb.pp("self_attn.q_norm"))?,
            k_norm: RmsNorm::new(head_dim, 1e-6, vb.pp("self_attn.k_norm"))?,
            n_heads, n_kv_heads, head_dim, rotary_dim, is_global, sliding_window, rope_theta,
        })
    }

    fn compute_kv(&self, x: &Tensor, pos: &Tensor) -> CResult<(Tensor, Tensor)> {
        let (b, seq, _) = x.dims3()?;
        let k = self.k_proj.forward(x)?
            .reshape((b, seq, self.n_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = self.v_proj.forward(x)?
            .reshape((b, seq, self.n_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let k = self.k_norm.forward(&k)?;
        let k = apply_rope(&k, pos, self.head_dim, self.rotary_dim, self.rope_theta)?;
        Ok((k, v))
    }

    fn forward(&self, x: &Tensor, pos: &Tensor, cached_kv: Option<&(Tensor, Tensor)>) -> CResult<Tensor> {
        let (b, seq, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?
            .reshape((b, seq, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        let q = self.q_norm.forward(&q)?;
        let q = apply_rope(&q, pos, self.head_dim, self.rotary_dim, self.rope_theta)?;

        let (k, v) = if let Some(kv) = cached_kv {
            (kv.0.clone(), kv.1.clone())
        } else {
            self.compute_kv(x, pos)?
        };

        let k = repeat_kv(k, self.n_heads / self.n_kv_heads)?;
        let v = repeat_kv(v, self.n_heads / self.n_kv_heads)?;

        let scaling = 1.0 / (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scaling)?;
        let attn = apply_causal_mask(attn, seq, if self.is_global { None } else { Some(self.sliding_window) })?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        let out = attn.matmul(&v)?
            .transpose(1, 2)?.reshape((b, seq, self.n_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }
}

fn apply_rope(x: &Tensor, pos: &Tensor, head_dim: usize, rotary_dim: usize, rope_theta: f64) -> CResult<Tensor> {
    let half = rotary_dim / 2;
    let (b, heads, seq, _) = x.dims4()?;
    let dev = x.device();
    let dtype = x.dtype();

    let freqs: Vec<f32> = (0..half)
        .map(|i| 1.0 / (rope_theta as f32).powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let freqs = Tensor::from_vec(freqs, (1, 1, 1, half), dev)?.to_dtype(dtype)?;
    let pos_f = pos.to_dtype(dtype)?.reshape((1, 1, seq, 1))?;
    let angles = pos_f.broadcast_mul(&freqs)?.broadcast_as((b, heads, seq, half))?;
    let cos = angles.cos()?;
    let sin = angles.sin()?;

    let x_rope = x.narrow(D::Minus1, 0, rotary_dim)?;
    let x1 = x_rope.narrow(D::Minus1, 0, half)?;
    let x2 = x_rope.narrow(D::Minus1, half, half)?;
    let x_rotated = Tensor::cat(&[
        x1.broadcast_mul(&cos)?.sub(&x2.broadcast_mul(&sin)?)?,
        x2.broadcast_mul(&cos)?.add(&x1.broadcast_mul(&sin)?)?,
    ], D::Minus1)?;

    if rotary_dim < head_dim {
        let x_pass = x.narrow(D::Minus1, rotary_dim, head_dim - rotary_dim)?;
        Tensor::cat(&[x_rotated, x_pass], D::Minus1)
    } else {
        Ok(x_rotated)
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> CResult<Tensor> {
    if n_rep == 1 { return Ok(x); }
    let (b, heads, seq, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, heads, n_rep, seq, d))?
        .reshape((b, heads * n_rep, seq, d))
}

fn apply_causal_mask(attn: Tensor, seq: usize, sliding_window: Option<usize>) -> CResult<Tensor> {
    let mask = (0..seq).map(|i| {
        (0..seq).map(|j| {
            if j > i || (sliding_window.is_some() && i >= sliding_window.unwrap() && j < i - sliding_window.unwrap()) {
                f32::NEG_INFINITY
            } else {
                0.0f32
            }
        }).collect::<Vec<_>>()
    }).flatten().collect::<Vec<_>>();
    let mask = Tensor::from_vec(mask, (seq, seq), attn.device())?.to_dtype(attn.dtype())?;
    attn.broadcast_add(&mask)
}

// ── MLP ──────────────────────────────────────────────────────────────────────

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn new(hidden: usize, intermediate: usize, vb: VarBuilder) -> CResult<Self> {
        Ok(Self {
            gate: linear(hidden, intermediate, vb.pp("mlp.gate_proj"))?,
            up: linear(hidden, intermediate, vb.pp("mlp.up_proj"))?,
            down: linear(intermediate, hidden, vb.pp("mlp.down_proj"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let gate = gelu_pytorch_tanh(&self.gate.forward(x)?)?;
        let up = self.up.forward(x)?;
        self.down.forward(&(gate * up)?)
    }
}

fn gelu_pytorch_tanh(x: &Tensor) -> CResult<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let sqrt_2_over_pi = (2.0f64 / std::f64::consts::PI).sqrt();
    let inner = (sqrt_2_over_pi * (x_f32.clone() + x_f32.powf(3.0)? * 0.044715)?)?;
    let res = ((x_f32 * 0.5)? * (inner.tanh()? + 1.0)?)?;
    res.to_dtype(x.dtype())
}

// ── Layer ─────────────────────────────────────────────────────────────────────

struct Layer {
    attn: Attention,
    mlp: Mlp,
    input_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    pre_ffn_norm: RmsNorm,
    post_ffn_norm: RmsNorm,
    per_layer_proj: Linear,
    per_layer_gate: Linear,
    per_layer_norm: RmsNorm,
    layer_scalar: Tensor,
}
impl Layer {
    fn new(vb: VarBuilder, hidden: usize, intermediate: usize,
           n_heads: usize, n_kv_heads: usize, head_dim: usize,
           per_layer_dim: usize, is_global: bool, sliding_window: usize,
           is_kv_shared: bool) -> CResult<Self> {
        let rope_theta = if is_global { 1_000_000.0f64 } else { 10_000.0f64 };
        Ok(Self {
            attn: Attention::new(vb.clone(), hidden, n_heads, n_kv_heads, head_dim, is_global, sliding_window, rope_theta, is_kv_shared)?,
            mlp: Mlp::new(hidden, intermediate, vb.clone())?,
            input_norm: RmsNorm::new(hidden, 1e-6, vb.pp("input_layernorm"))?,
            post_attn_norm: RmsNorm::new(hidden, 1e-6, vb.pp("post_attention_layernorm"))?,
            pre_ffn_norm: RmsNorm::new(hidden, 1e-6, vb.pp("pre_feedforward_layernorm"))?,
            post_ffn_norm: RmsNorm::new(hidden, 1e-6, vb.pp("post_feedforward_layernorm"))?,
            per_layer_proj: linear(per_layer_dim, hidden, vb.pp("per_layer_projection"))?,
            per_layer_gate: linear(hidden, per_layer_dim, vb.pp("per_layer_input_gate"))?,
            per_layer_norm: RmsNorm::new(hidden, 1e-6, vb.pp("post_per_layer_input_norm"))?,
            layer_scalar: vb.get(1, "layer_scalar")?,
        })
    }

    fn forward_with_kv(&self, x: &Tensor, pos: &Tensor, per_layer_emb: &Tensor,
               _cached_kv: Option<&(Tensor, Tensor)>) -> CResult<(Tensor, Option<(Tensor, Tensor)>)> {
        let scalar = self.layer_scalar.to_dtype(x.dtype())?.exp()?;

        let normed = self.input_norm.forward(x)?;
        let own_kv = self.attn.compute_kv(&normed, pos)?;
        let residual = x.clone();
        let h = self.attn.forward(&normed, pos, Some(&own_kv))?;
        let h = self.post_attn_norm.forward(&h)?;
        let x = (residual + h.broadcast_mul(&scalar)?)?;

        let residual = x.clone();
        let h = self.mlp.forward(&self.pre_ffn_norm.forward(&x)?)?;
        let h = self.post_ffn_norm.forward(&h)?;
        let x = (residual + h.broadcast_mul(&scalar)?)?;

        let residual = x.clone();
        let gate = gelu_pytorch_tanh(&self.per_layer_gate.forward(&x)?)?;
        let gated = (gate * per_layer_emb)?;
        let proj = self.per_layer_proj.forward(&gated)?;
        let per_layer = self.per_layer_norm.forward(&proj)?;
        let x = (residual + per_layer.broadcast_mul(&scalar)?)?;

        Ok((x, Some(own_kv)))
    }
}

// ── Model ─────────────────────────────────────────────────────────────────────

struct Gemma4TextModel {
    embed: candle_nn::Embedding,
    embed_per_layer: candle_nn::Embedding,
    per_layer_model_proj: Linear,
    layers: Vec<Layer>,
    norm: RmsNorm,
    per_layer_proj_norm: RmsNorm,
    lm_head_weight: Tensor,
    n_layers: usize,
    per_layer_dim: usize,
    final_logit_softcap: f64,
}

impl Gemma4TextModel {
    fn load(vb: VarBuilder, cfg: &serde_json::Value) -> CResult<Self> {
        let text = &cfg["text_config"];
        let hidden = text["hidden_size"].as_u64().unwrap_or(1536) as usize;
        let intermediate = text["intermediate_size"].as_u64().unwrap_or(6144) as usize;
        let n_heads = text["num_attention_heads"].as_u64().unwrap_or(8) as usize;
        let n_kv_heads = text["num_key_value_heads"].as_u64().unwrap_or(1) as usize;
        let head_dim_sliding = text["head_dim"].as_u64().unwrap_or(256) as usize;
        let head_dim_global = text["global_head_dim"].as_u64().unwrap_or(512) as usize;
        let n_layers = text["num_hidden_layers"].as_u64().unwrap_or(35) as usize;
        let vocab_size = text["vocab_size"].as_u64().unwrap_or(262144) as usize;
        let per_layer_dim = text["hidden_size_per_layer_input"].as_u64().unwrap_or(256) as usize;
        let sliding_window = text["sliding_window"].as_u64().unwrap_or(512) as usize;
        let final_logit_softcap = text["final_logit_softcapping"].as_f64().unwrap_or(30.0);
        let layer_types: Vec<String> = serde_json::from_value(text["layer_types"].clone()).unwrap_or_else(|_| vec!["sliding_attention".to_string(); n_layers]);
        let lm_vb = vb.pp("model.language_model");
        let embed = candle_nn::embedding(vocab_size, hidden, lm_vb.pp("embed_tokens"))?;
        let embed_per_layer = candle_nn::embedding(vocab_size, n_layers * per_layer_dim, lm_vb.pp("embed_tokens_per_layer"))?;
        let per_layer_model_proj = linear(hidden, n_layers * per_layer_dim, lm_vb.pp("per_layer_model_projection"))?;
        let num_kv_shared_layers = text["num_kv_shared_layers"].as_u64().unwrap_or(20) as usize;
        let double_wide_start = n_layers - num_kv_shared_layers;
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let is_global = layer_types.get(i).map_or(false, |t| t == "full_attention");
            let head_dim = if is_global { head_dim_global } else { head_dim_sliding };
            let layer_intermediate = if i >= double_wide_start { intermediate * 2 } else { intermediate };
            layers.push(Layer::new(lm_vb.pp(format!("layers.{i}")), hidden, layer_intermediate, n_heads, n_kv_heads, head_dim, per_layer_dim, is_global, sliding_window, false)?);
        }
        let norm = RmsNorm::new(hidden, 1e-6, lm_vb.pp("norm"))?;
        let per_layer_proj_norm = RmsNorm::new(per_layer_dim, 1e-6, lm_vb.pp("per_layer_projection_norm"))?;
        let lm_head_weight = lm_vb.get((vocab_size, hidden), "embed_tokens.weight")?;
        Ok(Self { embed, embed_per_layer, per_layer_model_proj, layers, norm, per_layer_proj_norm, lm_head_weight, n_layers, per_layer_dim, final_logit_softcap })
    }

    fn forward(&self, input_ids: &Tensor, pos: &Tensor) -> CResult<Tensor> {
        let (b, seq) = input_ids.dims2()?;
        let mut x = self.embed.forward(input_ids)?;
        // x = (x * (1536.0f64).sqrt())?; 
        let embed_part = (self.embed_per_layer.forward(input_ids)? * (self.per_layer_dim as f64).sqrt())?;
        let proj_part = (self.per_layer_model_proj.forward(&x)? * (1536.0f64).powf(-0.5))?;
        let proj_reshaped = proj_part.reshape((b, seq, self.n_layers, self.per_layer_dim))?;
        let proj_normed = self.per_layer_proj_norm.forward(&proj_reshaped)?;
        let proj_normed = proj_normed.reshape((b, seq, self.n_layers * self.per_layer_dim))?;
        let per_layer_all = ((proj_normed + embed_part)? * 2.0f64.powf(-0.5))?;
        for (i, layer) in self.layers.iter().enumerate() {
            let per_layer_emb = per_layer_all.narrow(D::Minus1, i * self.per_layer_dim, self.per_layer_dim)?;
            let (new_x, _) = layer.forward_with_kv(&x, pos, &per_layer_emb, None)?;
            x = new_x;
        }
        let x = self.norm.forward(&x)?;
        let (b, seq, h) = x.dims3()?;
        let x_2d = x.reshape((b * seq, h))?;
        let logits_2d = x_2d.matmul(&self.lm_head_weight.t()?)?;
        let logits_2d = (logits_2d * (1.0 / (1536.0f64).sqrt()))?;
        let logits = logits_2d.reshape((b, seq, self.lm_head_weight.dim(0)?))?;
        let cap = self.final_logit_softcap as f32;
        let logits = ((logits / cap as f64)?.tanh()? * cap as f64)?;
        Ok(logits)
    }
}

pub struct CandleBackend {
    model: Mutex<Gemma4TextModel>,
    config: ModelConfig,
    pool: CandleKvPool,
}

impl CandleBackend {
    pub fn load(model_dir: &Path) -> Result<Self, EngineError> {
        let config_str = std::fs::read_to_string(model_dir.join("config.json")).map_err(|e| EngineError::Backend(e.to_string()))?;
        let cfg: serde_json::Value = serde_json::from_str(&config_str).map_err(|e| EngineError::Backend(e.to_string()))?;
        let text = &cfg["text_config"];
        let model_config = ModelConfig {
            num_layers: text["num_hidden_layers"].as_u64().unwrap_or(35) as usize,
            num_kv_heads: text["num_key_value_heads"].as_u64().unwrap_or(1) as usize,
            head_dim: text["head_dim"].as_u64().unwrap_or(256) as usize,
            vocab_size: text["vocab_size"].as_u64().unwrap_or(262144) as usize,
            is_moe: false, num_experts: None, top_k_experts: None,
            eos_token_id: text["eos_token_id"].as_u64().unwrap_or(1) as u32,
        };
        let device = Device::Cpu;
        let weights = model_dir.join("model.safetensors");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, &device).map_err(|e| EngineError::Backend(format!("load weights: {e}")))? };
        let model = Gemma4TextModel::load(vb, &cfg).map_err(|e| EngineError::Backend(format!("build model: {e}")))?;
        Ok(Self { model: Mutex::new(model), config: model_config, pool: CandleKvPool { total: 512, free: 512 } })
    }
}

impl BackendHandle for CandleBackend {
    fn forward(&self, batch: &Batch) -> BoxFuture<'_, Result<Logits, EngineError>> {
        let num_requests = batch.requests.len();
        let vocab_size = self.config.vocab_size;

        let result = (|| -> Result<Logits, EngineError> {
            let device = Device::Cpu;
            let mut all_data = Vec::with_capacity(num_requests * vocab_size);
            for req_arc in &batch.requests {
                let req = req_arc.lock();
                let mut full_seq = req.input_ids.clone();
                full_seq.extend_from_slice(&req.output_ids);
                let seq_len = full_seq.len();
                
                let ids = Tensor::from_vec(full_seq, (1, seq_len), &device).map_err(|e| EngineError::Backend(e.to_string()))?;
                let pos = Tensor::arange(0u32, seq_len as u32, &device).map_err(|e| EngineError::Backend(e.to_string()))?.reshape((1, seq_len)).map_err(|e| EngineError::Backend(e.to_string()))?;
                
                let logits_3d = self.model.lock().unwrap().forward(&ids, &pos).map_err(|e| EngineError::Backend(e.to_string()))?;
                let last_logits = logits_3d.narrow(1, seq_len - 1, 1).map_err(|e| EngineError::Backend(e.to_string()))?
                    .flatten_all().map_err(|e| EngineError::Backend(e.to_string()))?
                    .to_vec1::<f32>()
                    .map_err(|e| EngineError::Backend(e.to_string()))?;
                all_data.extend_from_slice(&last_logits);
            }
            Ok(Logits { data: all_data, num_rows: num_requests, vocab_size })
        })();
        Box::pin(async move { result })
    }
    fn kv_pool(&self) -> &dyn KvPool { &self.pool }
    fn model_config(&self) -> &ModelConfig { &self.config }
}

struct CandleKvPool { total: usize, free: usize }
impl KvPool for CandleKvPool {
    fn free_pages(&self) -> usize { self.free }
    fn total_pages(&self) -> usize { self.total }
}
