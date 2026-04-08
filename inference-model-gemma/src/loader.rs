/// Weight loader for Gemma 4 safetensors format.
/// Phase 0: skeleton — maps HuggingFace weight names to Burn parameter paths.
use std::path::Path;

#[derive(Debug)]
pub struct LoadError(pub String);

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "load error: {}", self.0)
    }
}

/// Map a HuggingFace safetensors weight name to the Burn parameter path.
/// e.g. "model.layers.0.self_attn.q_proj.weight" -> "layers.0.attention.q_proj.weight"
pub fn hf_name_to_burn_path(hf_name: &str) -> Option<String> {
    let name = hf_name
        .strip_prefix("model.language_model.")?
        .strip_prefix("model.")?;

    let mapped = name
        .replace("self_attn.", "attention.")
        .replace("mlp.", "mlp.")
        .replace("input_layernorm.", "attention_norm.")
        .replace("post_feedforward_layernorm.", "mlp_norm.")
        .replace("embed_tokens.", "embedding.");

    Some(mapped)
}

/// Load Gemma 4 weights from a safetensors file.
/// Returns a map of parameter name -> raw f32 data.
/// Phase 0: validates the file exists and is readable.
pub fn load_safetensors(path: &Path) -> Result<(), LoadError> {
    if !path.exists() {
        return Err(LoadError(format!("weights file not found: {}", path.display())));
    }
    // Full loading deferred to Phase 1 (WgpuBackendHandle integration).
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hf_name_mapping() {
        // Verify the prefix stripping works for a known pattern
        let result = hf_name_to_burn_path("model.language_model.model.layers.0.self_attn.q_proj.weight");
        assert!(result.is_some());
        let path = result.unwrap();
        assert!(path.contains("attention.q_proj.weight"), "got: {path}");
    }
}
