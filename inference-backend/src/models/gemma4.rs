use candle_core::{Result as CResult, Tensor, D, Module};
use candle_nn::{linear_no_bias as linear, Linear, VarBuilder};
use crate::models::common::{RmsNorm, swiglu, apply_rope, repeat_kv, apply_causal_mask, CacheState};

pub struct Gemma4Attention {
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

impl Gemma4Attention {
    pub fn new(vb: VarBuilder, hidden: usize, n_heads: usize, n_kv_heads: usize,
           head_dim: usize, is_global: bool, sliding_window: usize, rope_theta: f64) -> CResult<Self> {
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let rotary_dim = if is_global { (head_dim as f32 * 0.25) as usize } else { head_dim };
        Ok(Self {
            q_proj: linear(hidden, q_dim, vb.pp("q_proj"))?,
            k_proj: linear(hidden, kv_dim, vb.pp("k_proj"))?,
            v_proj: linear(hidden, kv_dim, vb.pp("v_proj"))?,
            o_proj: linear(q_dim, hidden, vb.pp("o_proj"))?,
            q_norm: RmsNorm::new(head_dim, 1e-6, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(head_dim, 1e-6, vb.pp("k_norm"))?,
            n_heads, n_kv_heads, head_dim, rotary_dim, is_global, sliding_window, rope_theta,
        })
    }

    pub fn forward(&self, x: &Tensor, pos: &Tensor, k_cache: &Tensor, v_cache: &Tensor) -> CResult<(Tensor, Tensor, Tensor)> {
        let (b, seq, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?.reshape((b, seq, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        let q = self.q_norm.forward(&q)?;
        let q = apply_rope(&q, pos, self.head_dim, self.rotary_dim, self.rope_theta)?;

        let k = self.k_proj.forward(x)?.reshape((b, seq, self.n_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = self.v_proj.forward(x)?.reshape((b, seq, self.n_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let k = self.k_norm.forward(&k)?;
        let k = apply_rope(&k, pos, self.head_dim, self.rotary_dim, self.rope_theta)?;

        let k = Tensor::cat(&[k_cache.clone(), k], 2)?;
        let v = Tensor::cat(&[v_cache.clone(), v], 2)?;

        let k_ext = repeat_kv(k.clone(), self.n_heads / self.n_kv_heads)?;
        let v_ext = repeat_kv(v.clone(), self.n_heads / self.n_kv_heads)?;

        let scaling = 1.0 / (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k_ext.transpose(D::Minus2, D::Minus1)?)? * scaling)?;
        let attn_cap = 50.0f64;
        let attn = ((attn / attn_cap)?.tanh()? * attn_cap)?;

        let total_seq = k_ext.dim(2)?;
        let attn = apply_causal_mask(attn, seq, total_seq, if self.is_global { None } else { Some(self.sliding_window) })?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        let out = attn.matmul(&v_ext)?.transpose(1, 2)?.reshape((b, seq, ()))?;
        let out = self.o_proj.forward(&out)?;
        Ok((out, k, v))
    }
}

pub struct Gemma4Layer {
    pub attn: Gemma4Attention,
    pub ffn_gate: Linear,
    pub ffn_up: Linear,
    pub ffn_down: Linear,
    pub input_norm: RmsNorm,
    pub post_attn_norm: RmsNorm,
    pub pre_ffn_norm: RmsNorm,
    pub post_ffn_norm: RmsNorm,
    pub per_layer_proj: Linear,
    pub per_layer_gate: Linear,
    pub per_layer_norm: RmsNorm,
    pub layer_scalar: Tensor,
}

impl Gemma4Layer {
    pub fn forward(&self, x: &Tensor, pos: &Tensor, per_layer_emb: &Tensor, cache: &mut CacheState) -> CResult<Tensor> {
        let CacheState::Attention(ref mut k_cache, ref mut v_cache) = cache else { unreachable!() };
        let scalar = (self.layer_scalar.to_dtype(x.dtype())? + 1.0)?; 

        let normed = self.input_norm.forward(x)?;
        let residual = x.clone();
        let (h, nk, nv) = self.attn.forward(&normed, pos, k_cache, v_cache)?;
        *k_cache = nk; *v_cache = nv;
        let h = self.post_attn_norm.forward(&h)?;
        let x = (residual + h.broadcast_mul(&scalar)?)?;

        let residual = x.clone();
        let h = swiglu(&self.pre_ffn_norm.forward(&x)?, &self.ffn_gate, &self.ffn_up)?;
        let h = self.post_ffn_norm.forward(&h)?;
        let x = (residual + h.broadcast_mul(&scalar)?)?;

        let residual = x.clone();
        let gate = candle_nn::ops::sigmoid(&self.per_layer_gate.forward(&x)?)?;
        let gated = (gate * per_layer_emb)?;
        let proj = self.per_layer_proj.forward(&gated)?;
        let per_layer = self.per_layer_norm.forward(&proj)?;
        residual + per_layer.broadcast_mul(&scalar)?
    }
}

pub struct Gemma4TextModel {
    pub embed: candle_nn::Embedding,
    pub embed_per_layer: candle_nn::Embedding,
    pub per_layer_model_proj: Linear,
    pub layers: Vec<Gemma4Layer>,
    pub norm: RmsNorm,
    pub per_layer_proj_norm: RmsNorm,
    pub lm_head_weight: Tensor,
    pub vocab_size: usize,
    pub n_layers: usize,
    pub per_layer_dim: usize,
    pub final_logit_softcap: f64,
}

impl Gemma4TextModel {
    pub fn load(vb: VarBuilder, cfg: &serde_json::Value) -> CResult<Self> {
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
        
        let vb = vb.pp("model.language_model");
        let embed = candle_nn::embedding(vocab_size, hidden, vb.pp("embed_tokens"))?;
        let embed_per_layer = candle_nn::embedding(vocab_size, n_layers * per_layer_dim, vb.pp("embed_tokens_per_layer"))?;
        let per_layer_model_proj = linear(hidden, n_layers * per_layer_dim, vb.pp("per_layer_model_projection"))?;
        
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let is_global = layer_types.get(i).map_or(false, |t| t == "full_attention");
            let head_dim = if is_global { head_dim_global } else { head_dim_sliding };
            let l_vb = vb.pp(format!("layers.{}", i));
            layers.push(Gemma4Layer {
                attn: Gemma4Attention::new(l_vb.clone(), hidden, n_heads, n_kv_heads, head_dim, is_global, sliding_window, if is_global { 1000000.0 } else { 10000.0 })?,
                ffn_gate: linear(hidden, intermediate, l_vb.pp("mlp.gate_proj"))?,
                ffn_up: linear(hidden, intermediate, l_vb.pp("mlp.up_proj"))?,
                ffn_down: linear(intermediate, hidden, l_vb.pp("mlp.down_proj"))?,
                input_norm: RmsNorm::new(hidden, 1e-6, l_vb.pp("input_layernorm"))?,
                post_attn_norm: RmsNorm::new(hidden, 1e-6, l_vb.pp("post_attention_layernorm"))?,
                pre_ffn_norm: RmsNorm::new(hidden, 1e-6, l_vb.pp("pre_feedforward_layernorm"))?,
                post_ffn_norm: RmsNorm::new(hidden, 1e-6, l_vb.pp("post_feedforward_layernorm"))?,
                per_layer_proj: linear(per_layer_dim, hidden, l_vb.pp("per_layer_projection"))?,
                per_layer_gate: linear(hidden, per_layer_dim, l_vb.pp("per_layer_input_gate"))?,
                per_layer_norm: RmsNorm::new(hidden, 1e-6, l_vb.pp("post_per_layer_input_norm"))?,
                layer_scalar: l_vb.get(1, "layer_scalar")?,
            });
        }
        let norm = RmsNorm::new(hidden, 1e-6, vb.pp("norm"))?;
        let per_layer_proj_norm = RmsNorm::new(per_layer_dim, 1e-6, vb.pp("per_layer_projection_norm"))?;
        let lm_head_weight = vb.get((vocab_size, hidden), "embed_tokens.weight")?;
        Ok(Self { embed, embed_per_layer, per_layer_model_proj, layers, norm, per_layer_proj_norm, lm_head_weight, vocab_size, n_layers, per_layer_dim, final_logit_softcap })
    }

    pub fn forward(&self, input_ids: &Tensor, pos: &Tensor, kv_cache: &mut [CacheState]) -> CResult<Tensor> {
        let (b, seq) = input_ids.dims2()?;
        let mut x = self.embed.forward(input_ids)?;
        x = (x * (1536.0f64).sqrt())?;
        
        let embed_part = (self.embed_per_layer.forward(input_ids)? * (self.per_layer_dim as f64).sqrt())?;
        let proj_part = (self.per_layer_model_proj.forward(&x)? * (1536.0f64).powf(-0.5))?;
        let proj_reshaped = proj_part.reshape((b, seq, self.n_layers, self.per_layer_dim))?;
        let proj_normed = self.per_layer_proj_norm.forward(&proj_reshaped)?;
        let proj_normed = proj_normed.reshape((b, seq, self.n_layers * self.per_layer_dim))?;
        let per_layer_all = ((proj_normed + embed_part)? * 2.0f64.powf(-0.5))?;

        for (i, layer) in self.layers.iter().enumerate() {
            let per_layer_emb = per_layer_all.narrow(D::Minus1, i * self.per_layer_dim, self.per_layer_dim)?;
            x = layer.forward(&x, pos, &per_layer_emb, &mut kv_cache[i])?;
        }

        let x = self.norm.forward(&x)?;
        let (b, seq, h) = x.dims3()?;
        let x_2d = x.reshape((b * seq, h))?;
        let logits_2d = x_2d.matmul(&self.lm_head_weight.t()?)?;
        
        let logits_2d = (logits_2d * (1.0 / (h as f64).sqrt()))?;
        let cap = self.final_logit_softcap as f64;
        let logits_2d = (logits_2d / cap)?.tanh()?;
        let logits_2d = (logits_2d * cap)?;
        Ok(logits_2d.reshape((b, seq, self.vocab_size))?)
    }
}
