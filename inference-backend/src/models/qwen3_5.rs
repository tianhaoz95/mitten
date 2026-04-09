use candle_core::{DType, Result as CResult, Tensor, D, Module};
use candle_nn::{linear_no_bias as linear, Linear, VarBuilder, Conv1d, Conv1dConfig};
use crate::models::common::{RmsNorm, swiglu, apply_rope, repeat_kv, apply_causal_mask, l2norm, CacheState};

pub struct GatedDeltaNet {
    in_proj_qkv: Linear,
    in_proj_a: Linear,
    in_proj_b: Linear,
    in_proj_z: Linear,
    conv1d: Conv1d,
    norm: RmsNorm,
    out_proj: Linear,
    dt_bias: Tensor,
    a_log: Tensor,
    n_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
}

impl GatedDeltaNet {
    pub fn new(vb: VarBuilder, hidden: usize, n_heads: usize, head_k_dim: usize, head_v_dim: usize) -> CResult<Self> {
        let key_dim = n_heads * head_k_dim;
        let value_dim = n_heads * head_v_dim;
        let conv_dim = key_dim * 2 + value_dim;
        
        let conv_config = Conv1dConfig {
            groups: conv_dim,
            padding: 0,
            ..Default::default()
        };

        Ok(Self {
            in_proj_qkv: linear(hidden, conv_dim, vb.pp("in_proj_qkv"))?,
            in_proj_a: linear(hidden, n_heads, vb.pp("in_proj_a"))?,
            in_proj_b: linear(hidden, n_heads, vb.pp("in_proj_b"))?,
            in_proj_z: linear(hidden, value_dim, vb.pp("in_proj_z"))?,
            conv1d: candle_nn::conv1d_no_bias(conv_dim, conv_dim, 4, conv_config, vb.pp("conv1d"))?,
            norm: RmsNorm::new(head_v_dim, 1e-6, vb.pp("norm"))?,
            out_proj: linear(value_dim, hidden, vb.pp("out_proj"))?,
            dt_bias: vb.get(n_heads, "dt_bias")?,
            a_log: vb.get(n_heads, "A_log")?,
            n_heads,
            head_k_dim,
            head_v_dim,
        })
    }

    pub fn forward(&self, x: &Tensor, s: &Tensor, conv_cache: &Tensor) -> CResult<(Tensor, Tensor, Tensor)> {
        let (b, seq, _) = x.dims3()?;
        
        let mixed_qkv = self.in_proj_qkv.forward(x)?; 
        let a = self.in_proj_a.forward(x)?;
        let b_gate = self.in_proj_b.forward(x)?;
        let z = self.in_proj_z.forward(x)?;

        let qkv_conv_in = mixed_qkv.transpose(1, 2)?;
        let qkv_conv_in = Tensor::cat(&[conv_cache.clone(), qkv_conv_in], 2)?;
        let n_conv_cache = qkv_conv_in.narrow(2, qkv_conv_in.dim(2)? - 3, 3)?;

        let qkv_conv = self.conv1d.forward(&qkv_conv_in)?;
        let qkv_conv = qkv_conv.transpose(1, 2)?.silu()?; 

        let key_dim = self.n_heads * self.head_k_dim;
        let query = qkv_conv.narrow(2, 0, key_dim)?;
        let key = qkv_conv.narrow(2, key_dim, key_dim)?;
        let value = qkv_conv.narrow(2, 2 * key_dim, self.n_heads * self.head_v_dim)?;

        let q = query.reshape((b, seq, self.n_heads, self.head_k_dim))?;
        let k = key.reshape((b, seq, self.n_heads, self.head_k_dim))?;
        let v = value.reshape((b, seq, self.n_heads, self.head_v_dim))?;

        let q = l2norm(&q, 1e-6)?;
        let k = l2norm(&k, 1e-6)?;
        
        let scale = 1.0 / (self.head_k_dim as f64).sqrt();
        let q = (q * scale)?;

        let beta = candle_nn::ops::sigmoid(&b_gate)?;
        let a_plus_bias = a.broadcast_add(&self.dt_bias)?;
        let softplus_a = softplus(&a_plus_bias)?;
        let g = self.a_log.exp()?.broadcast_mul(&softplus_a)?.neg()?;
        let g_exp = g.exp()?;

        let mut s = s.clone();
        let mut outputs = Vec::with_capacity(seq);
        for t in 0..seq {
            let qt = q.narrow(1, t, 1)?.squeeze(1)?; 
            let kt = k.narrow(1, t, 1)?.squeeze(1)?; 
            let vt = v.narrow(1, t, 1)?.squeeze(1)?; 
            let gt = g_exp.narrow(1, t, 1)?.squeeze(1)?; 
            let bt = beta.narrow(1, t, 1)?.squeeze(1)?; 

            let pred_v = s.matmul(&kt.unsqueeze(D::Minus1)?)?.squeeze(D::Minus1)?; 
            let delta = (vt - pred_v)?;
            let update = delta.unsqueeze(D::Minus1)?.matmul(&kt.unsqueeze(D::Minus2)?)?; 
            
            s = s.broadcast_mul(&gt.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?)?;
            s = (s + update.broadcast_mul(&bt.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?)?)?;

            let yt = s.matmul(&qt.unsqueeze(D::Minus1)?)?.squeeze(D::Minus1)?; 
            outputs.push(yt.unsqueeze(1)?);
        }

        let y = Tensor::cat(&outputs, 1)?; // [b, seq, h, dv]
        
        let y_reshaped = y.reshape((b * seq * self.n_heads, self.head_v_dim))?;
        let y_normed = self.norm.forward(&y_reshaped)?;
        let y_normed = y_normed.reshape((b, seq, self.n_heads * self.head_v_dim))?;
        
        let out = (y_normed * z.silu()?)?;
        
        let out = self.out_proj.forward(&out)?;
        Ok((out, s, n_conv_cache))
    }
}

fn softplus(x: &Tensor) -> CResult<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let res = (x_f32.exp()? + 1.0)?.log()?;
    res.to_dtype(x.dtype())
}

pub struct GatedAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scaling: f64,
    rotary_dim: usize,
    rope_theta: f64,
}

impl GatedAttention {
    pub fn new(vb: VarBuilder, hidden: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, rope_theta: f64) -> CResult<Self> {
        Ok(Self {
            q_proj: linear(hidden, 2 * n_heads * head_dim, vb.pp("q_proj"))?,
            k_proj: linear(hidden, n_kv_heads * head_dim, vb.pp("k_proj"))?,
            v_proj: linear(hidden, n_kv_heads * head_dim, vb.pp("v_proj"))?,
            o_proj: linear(n_heads * head_dim, hidden, vb.pp("o_proj"))?,
            q_norm: RmsNorm::new_qwen(head_dim, 1e-6, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new_qwen(head_dim, 1e-6, vb.pp("k_norm"))?,
            n_heads,
            n_kv_heads,
            head_dim,
            scaling: (head_dim as f64).powf(-0.5),
            rotary_dim: (head_dim as f32 * 0.25) as usize,
            rope_theta,
        })
    }

    pub fn forward(&self, x: &Tensor, pos: &Tensor, k_cache: &Tensor, v_cache: &Tensor) -> CResult<(Tensor, Tensor, Tensor)> {
        let (b, seq, _) = x.dims3()?;
        
        let qz = self.q_proj.forward(x)?;
        let qz = qz.reshape((b, seq, self.n_heads, 2, self.head_dim))?;
        let q = qz.narrow(3, 0, 1)?.squeeze(3)?;
        let gate = qz.narrow(3, 1, 1)?.squeeze(3)?.reshape((b, seq, self.n_heads * self.head_dim))?;

        let k = self.k_proj.forward(x)?.reshape((b, seq, self.n_kv_heads, self.head_dim))?; // [b, seq, kv_h, d]
        let v = self.v_proj.forward(x)?.reshape((b, seq, self.n_kv_heads, self.head_dim))?;

        let q = self.q_norm.forward(&q)?.transpose(1, 2)?; // [b, h, seq, d]
        let k = self.k_norm.forward(&k)?.transpose(1, 2)?; // [b, kv_h, seq, d]
        let v = v.transpose(1, 2)?;

        let q = apply_rope(&q, pos, self.head_dim, self.rotary_dim, self.rope_theta)?;
        let k = apply_rope(&k, pos, self.head_dim, self.rotary_dim, self.rope_theta)?;

        let k = Tensor::cat(&[k_cache.clone(), k], 2)?;
        let v = Tensor::cat(&[v_cache.clone(), v], 2)?;

        let k_ext = repeat_kv(k.clone(), self.n_heads / self.n_kv_heads)?;
        let v_ext = repeat_kv(v.clone(), self.n_heads / self.n_kv_heads)?;

        let attn = (q.matmul(&k_ext.transpose(D::Minus2, D::Minus1)?)? * self.scaling)?;
        let total_seq = k_ext.dim(2)?;
        let attn = apply_causal_mask(attn, seq, total_seq, None)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        let out = attn.matmul(&v_ext)?.transpose(1, 2)?.reshape((b, seq, self.n_heads * self.head_dim))?;
        
        let out = (out * candle_nn::ops::sigmoid(&gate)?)?;
        
        let out = self.o_proj.forward(&out)?;
        Ok((out, k, v))
    }
}

pub struct Qwen3_5Layer {
    pub norm: RmsNorm,
    pub mixer: QwenMixer,
    pub ffn_norm: RmsNorm,
    pub ffn_gate: Linear,
    pub ffn_up: Linear,
    pub ffn_down: Linear,
}

pub enum QwenMixer {
    DeltaNet(GatedDeltaNet),
    Attention(GatedAttention),
}

impl Qwen3_5Layer {
    pub fn forward(&self, x: &Tensor, pos: &Tensor, cache: &mut CacheState) -> CResult<Tensor> {
        let residual = x.clone();
        let x = self.norm.forward(x)?;
        let x = match (&self.mixer, cache) {
            (QwenMixer::DeltaNet(m), CacheState::DeltaNet(ref mut s, ref mut c)) => {
                let (y, ns, nc) = m.forward(&x, s, c)?;
                *s = ns;
                *c = nc;
                y
            }
            (QwenMixer::Attention(m), CacheState::Attention(ref mut k, ref mut v)) => {
                let (y, nk, nv) = m.forward(&x, pos, k, v)?;
                *k = nk;
                *v = nv;
                y
            }
            _ => unreachable!("Cache state mismatch"),
        };
        let x = (residual + x)?;

        let residual = x.clone();
        let x = self.ffn_norm.forward(&x)?;
        let x = swiglu(&x, &self.ffn_gate, &self.ffn_up)?;
        let x = self.ffn_down.forward(&x)?;
        residual + x
    }
}

pub struct Qwen3_5Model {
    pub embed: candle_nn::Embedding,
    pub layers: Vec<Qwen3_5Layer>,
    pub norm: RmsNorm,
    pub lm_head: Linear,
    pub vocab_size: usize,
    pub hidden_size: usize,
}

impl Qwen3_5Model {
    pub fn load(vb: VarBuilder, cfg: &serde_json::Value) -> CResult<Self> {
        let text = &cfg["text_config"];
        let hidden = text["hidden_size"].as_u64().unwrap_or(1024) as usize;
        let intermediate = text["intermediate_size"].as_u64().unwrap_or(3584) as usize;
        let n_layers = text["num_hidden_layers"].as_u64().unwrap_or(24) as usize;
        let vocab_size = text["vocab_size"].as_u64().unwrap_or(248320) as usize;
        
        let vb_lm = vb.pp("model.language_model");

        let embed = candle_nn::embedding(vocab_size, hidden, vb_lm.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let l_vb = vb_lm.pp(format!("layers.{}", i));
            let mixer = if (i + 1) % 4 == 0 {
                QwenMixer::Attention(GatedAttention::new(l_vb.pp("self_attn"), hidden, 8, 2, 256, 10000000.0)?)
            } else {
                QwenMixer::DeltaNet(GatedDeltaNet::new(l_vb.pp("linear_attn"), hidden, 16, 128, 128)?)
            };
            layers.push(Qwen3_5Layer {
                norm: RmsNorm::new_qwen(hidden, 1e-6, l_vb.pp("input_layernorm"))?,
                mixer,
                ffn_norm: RmsNorm::new_qwen(hidden, 1e-6, l_vb.pp("post_attention_layernorm"))?,
                ffn_gate: linear(hidden, intermediate, l_vb.pp("mlp.gate_proj"))?,
                ffn_up: linear(hidden, intermediate, l_vb.pp("mlp.up_proj"))?,
                ffn_down: linear(intermediate, hidden, l_vb.pp("mlp.down_proj"))?,
            });
        }
        let norm = RmsNorm::new_qwen(hidden, 1e-6, vb_lm.pp("norm"))?;
        // tie_word_embeddings = true
        let lm_head = linear(hidden, vocab_size, vb_lm.pp("embed_tokens"))?;
        Ok(Self { embed, layers, norm, lm_head, vocab_size, hidden_size: hidden })
    }

    pub fn forward(&self, input_ids: &Tensor, pos: &Tensor, kv_cache: &mut [CacheState]) -> CResult<Tensor> {
        let (b, seq) = input_ids.dims2()?;
        let mut x = self.embed.forward(input_ids)?;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, pos, &mut kv_cache[i])?;
        }
        x = self.norm.forward(&x)?;
        let x_2d = x.reshape((b * seq, self.hidden_size))?;
        self.lm_head.forward(&x_2d)?.reshape((b, seq, self.vocab_size))
    }
}
