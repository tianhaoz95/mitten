use burn::prelude::*;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::module::Param;
use inference_model_common::{InferenceModel, KVCacheState};
use crate::config::Qwen3_5TextConfig;
use std::collections::HashMap;

#[derive(Module, Debug)]
pub struct Qwen3_5Model<B: Backend> {
    pub embedding: Embedding<B>,
    pub layers: Vec<Qwen3_5Layer<B>>,
    pub norm: RmsNorm<B>,
    pub output: MyLinear<B>,
    pub vocab_size: usize,
    pub hidden_size: usize,
}

#[derive(Module, Debug)]
pub enum QwenMixer<B: Backend> {
    DeltaNet(GatedDeltaNet<B>),
    Attention(GatedAttention<B>),
}

#[derive(Module, Debug)]
pub struct Qwen3_5Layer<B: Backend> {
    pub norm: RmsNorm<B>,
    pub mixer: QwenMixer<B>,
    pub ffn_norm: RmsNorm<B>,
    pub ffn_gate: MyLinear<B>,
    pub ffn_up: MyLinear<B>,
    pub ffn_down: MyLinear<B>,
}

#[derive(Module, Debug)]
pub struct GatedAttention<B: Backend> {
    pub q_proj: MyLinear<B>,
    pub k_proj: MyLinear<B>,
    pub v_proj: MyLinear<B>,
    pub o_proj: MyLinear<B>,
    pub q_norm: RmsNorm<B>,
    pub k_norm: RmsNorm<B>,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub scaling: f64,
    pub rope_theta: f64,
    pub rotary_dim: usize,
}

#[derive(Module, Debug)]
pub struct GatedDeltaNet<B: Backend> {
    pub in_proj_qkv: MyLinear<B>,
    pub in_proj_a: MyLinear<B>,
    pub in_proj_b: MyLinear<B>,
    pub in_proj_z: MyLinear<B>,
    pub conv1d: Conv1d<B>,
    pub norm: RmsNorm<B>,
    pub out_proj: MyLinear<B>,
    pub dt_bias: Param<Tensor<B, 1>>,
    pub a_log: Param<Tensor<B, 1>>,
    pub n_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
}

#[derive(Module, Debug)]
pub struct MyLinear<B: Backend> {
    pub weight: Param<Tensor<B, 2>>,
}

impl<B: Backend> MyLinear<B> {
    pub fn new(in_dim: usize, out_dim: usize, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::zeros([out_dim, in_dim], device)),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, _seq, _] = x.dims();
        let weight = self.weight.val().transpose();
        let [d1, d2] = weight.dims();
        let weight_3d = weight.reshape([1, d1, d2]).repeat(&[b, 1, 1]);
        x.matmul(weight_3d)
    }
}

#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    pub eps: f64,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(dim: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::zeros([dim], device)),
            eps,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let var = x.clone().powf_scalar(2.0).mean_dim(2);
        let x = x * (var + self.eps).sqrt().recip();
        x * (self.weight.val().reshape([1, 1, -1]) + 1.0)
    }
    
    pub fn forward4(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let var = x.clone().powf_scalar(2.0).mean_dim(3);
        let x = x * (var + self.eps).sqrt().recip();
        x * (self.weight.val().reshape([1, 1, 1, -1]) + 1.0)
    }
}

fn load_my_linear<B: Backend>(device: &B::Device, weights: &HashMap<String, Vec<f32>>, prefix: &str, in_dim: usize, out_dim: usize) -> MyLinear<B> {
    let mut linear = MyLinear::new(in_dim, out_dim, device);
    if let Some(w) = weights.get(&format!("{}.weight", prefix)) {
        linear.weight = Param::from_tensor(Tensor::<B, 2>::from_data(TensorData::new(w.clone(), [out_dim, in_dim]), device));
    }
    linear
}

fn load_embedding<B: Backend>(device: &B::Device, weights: &HashMap<String, Vec<f32>>, prefix: &str, vocab: usize, hidden: usize) -> Embedding<B> {
    let mut embedding = EmbeddingConfig::new(vocab, hidden).init(device);
    if let Some(w) = weights.get(&format!("{}.weight", prefix)) {
        let tensor = Tensor::<B, 2>::from_data(TensorData::new(w.clone(), [vocab, hidden]), device);
        embedding.weight = Param::from_tensor(tensor);
    }
    embedding
}

fn load_rms_norm<B: Backend>(device: &B::Device, weights: &HashMap<String, Vec<f32>>, prefix: &str, dim: usize, eps: f64) -> RmsNorm<B> {
    let mut norm = RmsNorm::new(dim, eps, device);
    if let Some(w) = weights.get(&format!("{}.weight", prefix)) {
        let tensor = Tensor::<B, 1>::from_data(TensorData::new(w.clone(), [dim]), device);
        norm.weight = Param::from_tensor(tensor);
    }
    norm
}

fn load_conv1d<B: Backend>(device: &B::Device, weights: &HashMap<String, Vec<f32>>, prefix: &str, channels: usize, kernel_size: usize) -> Conv1d<B> {
    let config = Conv1dConfig::new(channels, channels, kernel_size).with_bias(false).with_groups(channels);
    let mut conv = config.init(device);
    if let Some(w) = weights.get(&format!("{}.weight", prefix)) {
        let tensor = Tensor::<B, 3>::from_data(TensorData::new(w.clone(), [channels, 1, kernel_size]), device);
        conv.weight = Param::from_tensor(tensor);
    }
    conv
}

fn load_param1<B: Backend>(device: &B::Device, weights: &HashMap<String, Vec<f32>>, name: &str, dim: usize) -> Param<Tensor<B, 1>> {
    if let Some(w) = weights.get(name) {
        Param::from_tensor(Tensor::<B, 1>::from_data(TensorData::new(w.clone(), [dim]), device))
    } else {
        Param::from_tensor(Tensor::zeros([dim], device))
    }
}

impl<B: Backend> InferenceModel<B> for Qwen3_5Model<B> {
    type Config = Qwen3_5TextConfig;

    fn new(config: &Self::Config, device: &B::Device) -> Self {
        Self::new_with_weights(config, device, &HashMap::new())
    }

    fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        position_ids: Tensor<B, 1, Int>,
        kv_cache: &mut [KVCacheState<B>],
    ) -> Tensor<B, 3> {
        let mut x = self.embedding.forward(input_ids);

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x, position_ids.clone(), &mut kv_cache[i]);
        }

        let normed = self.norm.forward(x);
        let logits = self.output.forward(normed);
        logits
    }

    fn init_cache(&self, device: &B::Device) -> Vec<KVCacheState<B>> {
        self.layers.iter().map(|layer| {
            match &layer.mixer {
                QwenMixer::Attention(m) => KVCacheState::Attention(
                    Tensor::zeros([1, m.n_kv_heads, 0, m.head_dim], device),
                    Tensor::zeros([1, m.n_kv_heads, 0, m.head_dim], device),
                ),
                QwenMixer::DeltaNet(m) => KVCacheState::DeltaNet(
                    Tensor::zeros([1, m.n_heads, m.head_v_dim, m.head_k_dim], device),
                    Tensor::zeros([1, m.in_proj_qkv.weight.val().dims()[0], 3], device),
                ),
            }
        }).collect()
    }
}

impl<B: Backend> Qwen3_5Model<B> {
    pub fn new_with_weights(config: &Qwen3_5TextConfig, device: &B::Device, weights: &HashMap<String, Vec<f32>>) -> Self {
        let hidden = config.hidden_size;
        let vocab = config.vocab_size;
        let intermediate = config.intermediate_size;
        let eps = config.rms_norm_eps;
        let rope_theta = config.rope_theta();

        let embedding = load_embedding(device, weights, "embedding", vocab, hidden);
        let norm = load_rms_norm(device, weights, "norm", hidden, eps);
        let output = if weights.contains_key("output.weight") {
            load_my_linear(device, weights, "output", hidden, vocab)
        } else {
            let mut output = MyLinear::new(hidden, vocab, device);
            if let Some(w) = weights.get("embedding.weight") {
                 output.weight = Param::from_tensor(Tensor::<B, 2>::from_data(TensorData::new(w.clone(), [vocab, hidden]), device));
            }
            output
        };

        let layers = (0..config.num_hidden_layers).map(|i| {
            let prefix = format!("layers.{}", i);
            let mixer = if config.layer_type(i) == "full_attention" {
                let m_prefix = format!("{}.mixer", prefix);
                let n_heads = config.num_attention_heads;
                let n_kv_heads = config.num_key_value_heads;
                let head_dim = config.head_dim;
                QwenMixer::Attention(GatedAttention {
                    q_proj: load_my_linear(device, weights, &format!("{}.q_proj", m_prefix), hidden, n_heads * head_dim * 2),
                    k_proj: load_my_linear(device, weights, &format!("{}.k_proj", m_prefix), hidden, n_kv_heads * head_dim),
                    v_proj: load_my_linear(device, weights, &format!("{}.v_proj", m_prefix), hidden, n_kv_heads * head_dim),
                    o_proj: load_my_linear(device, weights, &format!("{}.o_proj", m_prefix), n_heads * head_dim, hidden),
                    q_norm: load_rms_norm(device, weights, &format!("{}.q_norm", m_prefix), head_dim, eps),
                    k_norm: load_rms_norm(device, weights, &format!("{}.k_norm", m_prefix), head_dim, eps),
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    scaling: (head_dim as f64).powf(-0.5),
                    rope_theta,
                    rotary_dim: config.rotary_dim(),
                })
            } else {
                let m_prefix = format!("{}.mixer", prefix);
                let n_heads = config.linear_num_key_heads;
                let head_k_dim = config.linear_key_head_dim;
                let head_v_dim = config.linear_value_head_dim;
                let qkv_dim = head_k_dim * n_heads * 2 + head_v_dim * n_heads;
                let z_dim = head_v_dim * n_heads;
                QwenMixer::DeltaNet(GatedDeltaNet {
                    in_proj_qkv: load_my_linear(device, weights, &format!("{}.in_proj_qkv", m_prefix), hidden, qkv_dim),
                    in_proj_a: load_my_linear(device, weights, &format!("{}.in_proj_a", m_prefix), hidden, n_heads),
                    in_proj_b: load_my_linear(device, weights, &format!("{}.in_proj_b", m_prefix), hidden, n_heads),
                    in_proj_z: load_my_linear(device, weights, &format!("{}.in_proj_z", m_prefix), hidden, z_dim),
                    conv1d: load_conv1d(device, weights, &format!("{}.conv1d", m_prefix), qkv_dim, 4),
                    norm: load_rms_norm(device, weights, &format!("{}.norm", m_prefix), head_k_dim, eps),
                    out_proj: load_my_linear(device, weights, &format!("{}.out_proj", m_prefix), z_dim, hidden),
                    dt_bias: load_param1(device, weights, &format!("{}.dt_bias", m_prefix), n_heads),
                    a_log: load_param1(device, weights, &format!("{}.A_log", m_prefix), n_heads),
                    n_heads,
                    head_k_dim,
                    head_v_dim,
                })
            };

            Qwen3_5Layer {
                norm: load_rms_norm(device, weights, &format!("{}.norm", prefix), hidden, eps),
                mixer,
                ffn_norm: load_rms_norm(device, weights, &format!("{}.ffn_norm", prefix), hidden, eps),
                ffn_gate: load_my_linear(device, weights, &format!("{}.mlp.gate_proj", prefix), hidden, intermediate),
                ffn_up: load_my_linear(device, weights, &format!("{}.mlp.up_proj", prefix), hidden, intermediate),
                ffn_down: load_my_linear(device, weights, &format!("{}.mlp.down_proj", prefix), intermediate, hidden),
            }
        }).collect();

        Self { embedding, layers, norm, output, vocab_size: vocab, hidden_size: hidden }
    }
}

impl<B: Backend> Qwen3_5Layer<B> {
    pub fn forward(&self, x: Tensor<B, 3>, position_ids: Tensor<B, 1, Int>, cache: &mut KVCacheState<B>) -> Tensor<B, 3> {
        let residual = x.clone();
        let normed = self.norm.forward(x);
        
        let mixed = match &self.mixer {
            QwenMixer::Attention(m) => m.forward(normed, position_ids, cache),
            QwenMixer::DeltaNet(m) => m.forward(normed, cache),
        };
        let x = residual + mixed;

        let residual = x.clone();
        let normed = self.ffn_norm.forward(x);
        let gate = self.ffn_gate.forward(normed.clone());
        let up = self.ffn_up.forward(normed);
        let ffn = self.ffn_down.forward(burn::tensor::activation::silu(gate) * up);
        
        residual + ffn
    }
}

impl<B: Backend> GatedAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>, position_ids: Tensor<B, 1, Int>, cache: &mut KVCacheState<B>) -> Tensor<B, 3> {
        let [b, seq, _] = x.dims();
        // q_proj output is [n_heads * head_dim * 2]: first half = q, second half = gate
        let qg = self.q_proj.forward(x.clone()).reshape([b, seq, self.n_heads, 2, self.head_dim]);
        let q = qg.clone().slice([0..b, 0..seq, 0..self.n_heads, 0..1, 0..self.head_dim]).squeeze::<4>(3);
        let gate = qg.slice([0..b, 0..seq, 0..self.n_heads, 1..2, 0..self.head_dim]).squeeze::<4>(3)
            .reshape([b, seq, self.n_heads * self.head_dim]);

        let k = self.k_proj.forward(x.clone()).reshape([b, seq, self.n_kv_heads, self.head_dim]);
        let v = self.v_proj.forward(x).reshape([b, seq, self.n_kv_heads, self.head_dim]);

        let q = self.q_norm.forward4(q);
        let k = self.k_norm.forward4(k);

        let q = apply_rope(q, position_ids.clone(), self.head_dim, self.rotary_dim, self.rope_theta);
        let k = apply_rope(k, position_ids, self.head_dim, self.rotary_dim, self.rope_theta);

        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        if let KVCacheState::Attention(ref mut k_cache, ref mut v_cache) = cache {
            *k_cache = Tensor::cat(vec![k_cache.clone(), k], 2);
            *v_cache = Tensor::cat(vec![v_cache.clone(), v], 2);

            let [_, heads, seq_q, _] = q.dims();
            let [_, n_kv_heads, total_seq, head_dim] = k_cache.dims();
            let n_rep = heads / n_kv_heads;

            let k_ext = if n_rep > 1 {
                k_cache.clone().reshape([b, n_kv_heads, 1, total_seq, head_dim])
                    .repeat(&[1, 1, n_rep, 1, 1])
                    .reshape([b, heads, total_seq, head_dim])
            } else { k_cache.clone() };

            let v_ext = if n_rep > 1 {
                v_cache.clone().reshape([b, n_kv_heads, 1, total_seq, head_dim])
                    .repeat(&[1, 1, n_rep, 1, 1])
                    .reshape([b, heads, total_seq, head_dim])
            } else { v_cache.clone() };

            let mask = generate_causal_mask::<B>(seq_q, total_seq, &q.device());
            let mask = mask.repeat(&[b, heads, 1, 1]);

            let attn = q.matmul(k_ext.transpose()) * self.scaling;
            let attn = burn::tensor::activation::softmax(attn + mask, 3);

            let out = attn.matmul(v_ext).swap_dims(1, 2).reshape([b, seq, self.n_heads * self.head_dim]);
            let out = burn::tensor::activation::sigmoid(gate) * out;
            self.o_proj.forward(out)
        } else {
            panic!("Cache type mismatch")
        }
    }
}

impl<B: Backend> GatedDeltaNet<B> {
    pub fn forward(&self, x: Tensor<B, 3>, cache: &mut KVCacheState<B>) -> Tensor<B, 3> {
        let [b, seq, _] = x.dims();
        
        let mixed_qkv = self.in_proj_qkv.forward(x.clone()); 
        let a = self.in_proj_a.forward(x.clone());
        let b_gate = self.in_proj_b.forward(x.clone());
        let z = self.in_proj_z.forward(x);

        let qkv_conv_in = mixed_qkv.swap_dims(1, 2); // [b, dim, seq]
        
        if let KVCacheState::DeltaNet(ref mut s, ref mut conv_cache) = cache {
            let qkv_conv_in = Tensor::cat(vec![conv_cache.clone(), qkv_conv_in], 2);
            *conv_cache = qkv_conv_in.clone().slice([0..b, 0..6144, (qkv_conv_in.dims()[2] - 3)..qkv_conv_in.dims()[2]]);
            
            let qkv_conv = self.conv1d.forward(qkv_conv_in);
            let qkv_conv = burn::tensor::activation::silu(qkv_conv.swap_dims(1, 2)); // [b, seq, 6144]

            let q = qkv_conv.clone().slice([0..b, 0..seq, 0 .. 2048]).reshape([b, seq, self.n_heads, self.head_k_dim]);
            let k = qkv_conv.clone().slice([0..b, 0..seq, 2048 .. 4096]).reshape([b, seq, self.n_heads, self.head_k_dim]);
            let v = qkv_conv.slice([0..b, 0..seq, 4096 .. 6144]).reshape([b, seq, self.n_heads, self.head_v_dim]);

            let q = l2norm(q, 1e-6) * (1.0 / (self.head_k_dim as f64).sqrt());
            let k = l2norm(k, 1e-6);
            
            let beta = burn::tensor::activation::sigmoid(b_gate).reshape([b, seq, self.n_heads, 1, 1]);
            let a_plus_bias = a + self.dt_bias.val().reshape([1, 1, -1]);
            let softplus_a = (a_plus_bias.exp() + 1.0).log();
            let g = (self.a_log.val().reshape([1, 1, -1]).exp() * softplus_a).neg().exp().reshape([b, seq, self.n_heads, 1, 1]);

            let current_s = s.clone();
            let mut head_outputs = Vec::with_capacity(self.n_heads);
            
            for h in 0..self.n_heads {
                let qh = q.clone().slice([0..b, 0..seq, h..h+1, 0..self.head_k_dim]).reshape([b, seq, 1, self.head_k_dim]);
                let kh = k.clone().slice([0..b, 0..seq, h..h+1, 0..self.head_k_dim]).reshape([b, seq, 1, self.head_k_dim]);
                let vh = v.clone().slice([0..b, 0..seq, h..h+1, 0..self.head_v_dim]).reshape([b, seq, 1, self.head_v_dim]);
                let gh = g.clone().slice([0..b, 0..seq, h..h+1, 0..1, 0..1]).reshape([b, seq, 1, 1]);
                let bh = beta.clone().slice([0..b, 0..seq, h..h+1, 0..1, 0..1]).reshape([b, seq, 1, 1]);
                
                let mut sh = current_s.clone().slice([0..b, h..h+1, 0..self.head_v_dim, 0..self.head_k_dim]);
                let mut outputs = Vec::with_capacity(seq);
                
                for t in 0..seq {
                    let qht = qh.clone().slice([0..b, t..t+1, 0..1, 0..self.head_k_dim]);
                    let kht = kh.clone().slice([0..b, t..t+1, 0..1, 0..self.head_k_dim]);
                    let vht = vh.clone().slice([0..b, t..t+1, 0..1, 0..self.head_v_dim]);
                    let ght = gh.clone().slice([0..b, t..t+1, 0..1, 0..1]);
                    let bht = bh.clone().slice([0..b, t..t+1, 0..1, 0..1]);
                    
                    let pred_v = sh.clone().matmul(kht.clone().reshape([b, 1, self.head_k_dim, 1])); 
                    let delta = vht.reshape([b, 1, self.head_v_dim, 1]) - pred_v;
                    let update = delta.matmul(kht.reshape([b, 1, 1, self.head_k_dim])) * bht;
                    
                    sh = sh * ght + update;
                    let yht = sh.clone().matmul(qht.reshape([b, 1, self.head_k_dim, 1]));
                    outputs.push(yht.reshape([b, 1, self.head_v_dim]));
                }
                head_outputs.push((Tensor::cat(outputs, 1), sh));
            }
            
            let y_heads: Vec<Tensor<B, 3>> = head_outputs.iter().map(|(y, _)| y.clone()).collect();
            let s_heads: Vec<Tensor<B, 4>> = head_outputs.iter().map(|(_, s)| s.clone()).collect();
            
            *s = Tensor::cat(s_heads, 1);
            let y = Tensor::cat(y_heads, 2); 
            
            let y_normed = self.norm.forward4(y.reshape([b, seq, self.n_heads, self.head_v_dim]));
            let y_normed = y_normed.reshape([b, seq, self.n_heads * self.head_v_dim]);
            
            let out = burn::tensor::activation::silu(z) * y_normed;
            self.out_proj.forward(out)
        } else {
            panic!("Cache type mismatch")
        }
    }
}

fn l2norm<B: Backend, const D: usize>(x: Tensor<B, D>, eps: f64) -> Tensor<B, D> {
    let var = x.clone().powf_scalar(2.0).sum_dim(D - 1);
    x * (var + eps).sqrt().recip()
}

fn apply_rope<B: Backend>(x: Tensor<B, 4>, pos: Tensor<B, 1, Int>, _head_dim: usize, rotary_dim: usize, rope_theta: f64) -> Tensor<B, 4> {
    let [b, seq, heads, x_head_dim] = x.dims();
    let half = rotary_dim / 2;

    let freqs: Vec<f32> = (0..half)
        .map(|i| 1.0 / (rope_theta as f32).powf(2.0 * i as f32 / rotary_dim as f32))
        .collect();
    
    let device = x.device();
    let freqs = Tensor::<B, 1>::from_data(TensorData::new(freqs, [half]), &device).reshape([1, 1, 1, half]);
    let pos_f = pos.float().reshape([1, seq, 1, 1]);
    
    let angles = pos_f.repeat(&[1, 1, 1, half]) * freqs.repeat(&[1, seq, 1, 1]);
    
    let cos = angles.clone().cos();
    let sin = angles.sin();

    let x_rope = x.clone().slice([0..b, 0..seq, 0..heads, 0..rotary_dim]);
    let x1 = x_rope.clone().slice([0..b, 0..seq, 0..heads, 0..half]);
    let x2 = x_rope.slice([0..b, 0..seq, 0..heads, half..rotary_dim]);

    let x_rotated = Tensor::cat(vec![
        x1.clone() * cos.clone() - x2.clone() * sin.clone(),
        x2 * cos + x1 * sin,
    ], 3);

    if rotary_dim < x_head_dim {
        let x_pass = x.slice([0..b, 0..seq, 0..heads, rotary_dim..x_head_dim]);
        Tensor::cat(vec![x_rotated, x_pass], 3)
    } else {
        x_rotated
    }
}

fn generate_causal_mask<B: Backend>(seq: usize, total_seq: usize, device: &B::Device) -> Tensor<B, 4> {
    let mut mask_data = Vec::with_capacity(seq * total_seq);
    for i in 0..seq {
        let cur_pos = total_seq - seq + i;
        for j in 0..total_seq {
            if j > cur_pos {
                mask_data.push(-1e10f32);
            } else {
                mask_data.push(0.0f32);
            }
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(mask_data, [seq, total_seq]), device).reshape([1, 1, seq, total_seq])
}
