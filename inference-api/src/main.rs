use inference_api::{create_router, AppState, start_tokenizer_service};
use inference_engine::engine_loop::{run_overlapped_loop, EngineContext};
use inference_engine::config::{EngineConfig, EngineStats};
use inference_engine::kv_pool::KvCachePool;
use inference_engine::radix_cache::RadixCache;
use inference_backend::backend::BackendHandle;
use inference_backend::burn_backend::BurnBackend;
use inference_model_qwen::model::Qwen3_5Model;
use inference_model_gemma::model::Gemma4Model;
use inference_model_common::InferenceModel;
use std::sync::Arc;
use tokio::sync::mpsc;
use std::path::PathBuf;

use burn::backend::{NdArray, Wgpu};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::ndarray::NdArrayDevice;
use burn::prelude::Backend;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let mut args = std::env::args().skip(1);
    let mut model_dir = PathBuf::from("model/gemma-4-E2B-it");
    let mut use_gpu = false;
    while let Some(arg) = args.next() {
        if arg == "--model-dir" {
            if let Some(path) = args.next() {
                model_dir = PathBuf::from(path);
            }
        } else if arg == "--use-gpu" {
            use_gpu = true;
        }
    }

    let tokenizer_path = model_dir.join("tokenizer.json");

    eprintln!(">> Loading weights from {} (GPU: {})...", model_dir.display(), use_gpu);
    
    let backend: Arc<dyn BackendHandle> = if use_gpu {
        load_backend::<Wgpu>(&model_dir, WgpuDevice::DefaultDevice)
    } else {
        load_backend::<NdArray>(&model_dir, NdArrayDevice::default())
    };

    eprintln!(">> Model loaded.");

    let model_config = backend.model_config().clone();
    let num_pages = 128;
    let page_size = 16;

    let engine_config = EngineConfig {
        max_total_tokens: 8192,
        num_kv_pages: num_pages,
        page_size,
        max_prefill_tokens_per_iter: 512,
        max_decode_batch_size: 8,
        model_config,
    };

    let tokenizer_for_engine = tokenizers::Tokenizer::from_file(&tokenizer_path).expect("Failed to load tokenizer for engine");
    let detokenizer = Arc::new(move |ids: &[u32]| {
        tokenizer_for_engine.decode(ids, true).unwrap_or_else(|_| format!("{:?}", ids))
    });

    let ctx = EngineContext {
        backend: backend.clone(),
        radix_cache: RadixCache::new(page_size, num_pages),
        kv_pool: KvCachePool::new(num_pages, page_size),
        waiting: std::collections::VecDeque::new(),
        prefilling: Vec::new(),
        decoding: Vec::new(),
        config: engine_config,
        stats: EngineStats::default(),
        detokenizer: Some(detokenizer),
    };

    let (sched_tx, sched_rx) = mpsc::channel(32);
    let (gpu_tx, gpu_rx) = std::sync::mpsc::sync_channel(1);

    tokio::spawn(run_overlapped_loop(ctx, sched_rx, gpu_tx));

    let backend_thread = backend.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        while let Ok(work) = gpu_rx.recv() {
            let res = rt.block_on(BackendHandle::forward(backend_thread.as_ref(), &work.batch));
            let _ = work.result_tx.send(res);
        }
    });

    let tokenizer = start_tokenizer_service(&tokenizer_path).expect("Failed to start tokenizer");

    let model_name = if model_dir.to_string_lossy().contains("qwen") {
        "qwen-3.5-0.8b"
    } else {
        "gemma-4-e2b-it"
    }.to_string();

    let state = AppState {
        engine_tx: sched_tx,
        tokenizer,
        model_name,
    };

    let app = create_router(state);
    let port = 8080;
    let listener = match tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await {
        Ok(l) => {
            eprintln!(">> API server listening on http://0.0.0.0:{}", port);
            l
        }
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
            let fallback_port = 8081;
            eprintln!(">> Port {} in use, trying fallback http://0.0.0.0:{}", port, fallback_port);
            tokio::net::TcpListener::bind(format!("0.0.0.0:{}", fallback_port)).await.expect("Failed to bind to both 8080 and 8081")
        }
        Err(e) => panic!("Failed to bind: {}", e),
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c().await.expect("failed to install CTRL+C handler");
            eprintln!(">> Shutting down...");
        })
        .await
        .unwrap();
}

fn load_backend<B: Backend + 'static>(model_dir: &std::path::Path, device: B::Device) -> Arc<dyn BackendHandle> 
{
    if model_dir.to_string_lossy().contains("qwen") {
        use inference_model_qwen::config::Qwen3_5Config;
        use inference_model_qwen::loader;
        use inference_backend::config::ModelConfig;

        let config_str = std::fs::read_to_string(model_dir.join("config.json")).expect("No config.json");
        let q_cfg: Qwen3_5Config = Qwen3_5Config::from_json(&config_str).expect("Failed to parse config");

        let weights_path = model_dir.join("model.safetensors");
        let weights_data = loader::load_safetensors_data(&weights_path).expect("Failed to load weights");

        let model = Qwen3_5Model::<B>::new_with_weights(&q_cfg.text_config, &device, &weights_data);

        let model_config = ModelConfig {
            num_layers: q_cfg.text_config.num_hidden_layers,
            num_kv_heads: q_cfg.text_config.num_key_value_heads,
            head_dim: q_cfg.text_config.head_dim,
            vocab_size: q_cfg.text_config.vocab_size,
            is_moe: false,
            num_experts: None,
            top_k_experts: None,
            eos_token_ids: vec![q_cfg.text_config.eos_token_id, 248046],
        };

        Arc::new(BurnBackend::new(model, model_config, device))
    } else {
        use inference_model_gemma::config::Gemma4Config;
        use inference_backend::config::ModelConfig;

        let config_str = std::fs::read_to_string(model_dir.join("config.json")).expect("No config.json");
        let g_cfg: Gemma4Config = Gemma4Config::from_json(&config_str).expect("Failed to parse config");

        let model = Gemma4Model::<B>::new(&g_cfg.text_config, &device);

        let model_config = ModelConfig {
            num_layers: g_cfg.text_config.num_hidden_layers,
            num_kv_heads: g_cfg.text_config.num_key_value_heads,
            head_dim: g_cfg.text_config.head_dim,
            vocab_size: g_cfg.text_config.vocab_size,
            is_moe: false,
            num_experts: None,
            top_k_experts: None,
            eos_token_ids: vec![g_cfg.text_config.eos_token_id, 106],
        };

        Arc::new(BurnBackend::new(model, model_config, device))
    }
}
