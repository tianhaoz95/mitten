use std::path::Path;
use std::collections::HashMap;
use safetensors::{SafeTensors, Dtype};
use memmap2::MmapOptions;

#[derive(Debug)]
pub struct LoadError(pub String);

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "load error: {}", self.0)
    }
}

impl std::error::Error for LoadError {}

/// Map a HuggingFace safetensors weight name to the Burn parameter path.
/// e.g. "model.language_model.model.layers.0.self_attn.q_proj.weight" -> "layers.0.mixer.q_proj.weight"
pub fn hf_name_to_burn_path(hf_name: &str) -> Option<String> {
    // Qwen 3.5 specific mapping
    let name = if let Some(stripped) = hf_name.strip_prefix("model.language_model.") {
        stripped
    } else if let Some(stripped) = hf_name.strip_prefix("model.") {
        stripped
    } else {
        return None;
    };

    let mapped = if name.contains("self_attn.") {
        name.replace("self_attn.", "mixer.")
            .replace("q_proj.", "q_proj.")
            .replace("k_proj.", "k_proj.")
            .replace("v_proj.", "v_proj.")
            .replace("o_proj.", "o_proj.")
            .replace("q_norm.", "q_norm.")
            .replace("k_norm.", "k_norm.")
    } else if name.contains("linear_attn.") {
        name.replace("linear_attn.", "mixer.")
    } else {
        name.to_string()
    };

    let mapped = mapped
        .replace("input_layernorm.", "norm.")
        .replace("post_attention_layernorm.", "ffn_norm.")
        .replace("embed_tokens.", "embedding.");

    Some(mapped)
}

/// Load weights from safetensors into a hashmap for Burn record loading.
pub fn load_safetensors_data(path: &Path) -> Result<HashMap<String, Vec<f32>>, LoadError> {
    let file = std::fs::File::open(path).map_err(|e| LoadError(e.to_string()))?;
    let mmap = unsafe { MmapOptions::new().map(&file).map_err(|e| LoadError(e.to_string()))? };
    let safetensors = SafeTensors::deserialize(&mmap).map_err(|e| LoadError(e.to_string()))?;

    let mut data = HashMap::new();
    for name in safetensors.names() {
        if let Some(burn_path) = hf_name_to_burn_path(name) {
            let tensor = safetensors.tensor(name).map_err(|e| LoadError(e.to_string()))?;
            let mut f32_data = Vec::with_capacity(tensor.data().len() / 4);
            
            // Assume F32 for now, or handle BF16/F16
            match tensor.dtype() {
                Dtype::F32 => {
                    for i in 0..(tensor.data().len() / 4) {
                        let bytes = &tensor.data()[i*4..(i+1)*4];
                        f32_data.push(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
                    }
                }
                Dtype::BF16 => {
                   for i in 0..(tensor.data().len() / 2) {
                        let bytes = &tensor.data()[i*2..(i+1)*2];
                        let val = half::bf16::from_le_bytes([bytes[0], bytes[1]]);
                        f32_data.push(val.to_f32());
                    }
                }
                Dtype::F16 => {
                   for i in 0..(tensor.data().len() / 2) {
                        let bytes = &tensor.data()[i*2..(i+1)*2];
                        let val = half::f16::from_le_bytes([bytes[0], bytes[1]]);
                        f32_data.push(val.to_f32());
                    }
                }
                _ => return Err(LoadError(format!("Unsupported dtype: {:?}", tensor.dtype()))),
            }
            data.insert(burn_path, f32_data);
        }
    }
    Ok(data)
}
