use candle_core::{DType, Result as CResult, Tensor, D, Module};
use candle_nn::{Linear, VarBuilder};

pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
    use_qwen_scaling: bool,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> CResult<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps, use_qwen_scaling: false })
    }
    
    pub fn new_qwen(dim: usize, eps: f64, vb: VarBuilder) -> CResult<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps, use_qwen_scaling: true })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let norm = (x_f32.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?.sqrt()?;
        let x_normed = x_f32.broadcast_div(&norm)?;
        
        let w_f32 = self.weight.to_dtype(DType::F32)?;
        let w = if self.use_qwen_scaling {
            (w_f32 + 1.0)?
        } else {
            w_f32
        };
        
        let res = x_normed.broadcast_mul(&w)?;
        res.to_dtype(x.dtype())
    }
}

pub fn swiglu(x: &Tensor, gate: &Linear, up: &Linear) -> CResult<Tensor> {
    let g = gate.forward(x)?;
    let g = candle_nn::ops::silu(&g)?;
    let u = up.forward(x)?;
    g * u
}

pub fn l2norm(x: &Tensor, eps: f64) -> CResult<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let norm = (x_f32.sqr()?.sum_keepdim(D::Minus1)? + eps)?.sqrt()?;
    let x_normed = x_f32.broadcast_div(&norm)?;
    x_normed.to_dtype(x.dtype())
}

pub fn apply_rope(x: &Tensor, pos: &Tensor, _head_dim: usize, rotary_dim: usize, rope_theta: f64) -> CResult<Tensor> {
    let half = rotary_dim / 2;
    let (b, heads, seq, x_head_dim) = x.dims4()?;
    let dev = x.device();
    let dtype = x.dtype();

    let freqs: Vec<f32> = (0..half)
        .map(|i| 1.0 / (rope_theta as f32).powf(2.0 * i as f32 / rotary_dim as f32))
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

    if rotary_dim < x_head_dim {
        let x_pass = x.narrow(D::Minus1, rotary_dim, x_head_dim - rotary_dim)?;
        Tensor::cat(&[x_rotated, x_pass], D::Minus1)
    } else {
        Ok(x_rotated)
    }
}

pub fn repeat_kv(x: Tensor, n_rep: usize) -> CResult<Tensor> {
    if n_rep == 1 { return Ok(x); }
    let (b, heads, seq, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, heads, n_rep, seq, d))?
        .reshape((b, heads * n_rep, seq, d))
}

pub fn apply_causal_mask(attn: Tensor, seq: usize, total_seq: usize, sliding_window: Option<usize>) -> CResult<Tensor> {
    let device = attn.device();
    let mask = (0..seq).map(|i| {
        let cur_pos = total_seq - seq + i;
        (0..total_seq).map(|j| {
            if j > cur_pos || (sliding_window.is_some() && cur_pos >= sliding_window.unwrap() && j < cur_pos - sliding_window.unwrap()) {
                f32::NEG_INFINITY
            } else {
                0.0f32
            }
        }).collect::<Vec<_>>()
    }).flatten().collect::<Vec<_>>();
    let mask = Tensor::from_vec(mask, (seq, total_seq), device)?.to_dtype(attn.dtype())?;
    attn.broadcast_add(&mask)
}

#[derive(Clone)]
pub enum CacheState {
    Attention(Tensor, Tensor),
    DeltaNet(Tensor, Tensor), // state s, and conv_cache
}
