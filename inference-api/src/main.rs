use inference_api::{create_router, AppState, start_tokenizer_service};
use inference_engine::engine_loop::{run_overlapped_loop, EngineContext};
use inference_engine::config::{EngineConfig, EngineStats};
use inference_engine::kv_pool::KvCachePool;
use inference_engine::radix_cache::RadixCache;
use inference_backend::backend::BackendHandle;
use inference_backend::{CandleBackend, burn_backend::BurnBackend};
use inference_model_qwen::model::Qwen3_5Model;
use inference_model_common::InferenceModel;
use burn::backend::NdArray; // Default CPU backend for now
use std::sync::Arc;
use tokio::sync::mpsc;
use std::path::PathBuf;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let mut args = std::env::args().skip(1);
    let mut model_dir = PathBuf::from("model/gemma-4-E2B-it");
    while let Some(arg) = args.next() {
        if arg == "--model-dir" {
            if let Some(path) = args.next() {
                model_dir = PathBuf::from(path);
            }
        }
    }

    let tokenizer_path = model_dir.join("tokenizer.json");

    eprintln!(">> Loading weights from {}...", model_dir.display());
    let backend: Arc<dyn BackendHandle> = if model_dir.to_string_lossy().contains("qwen") {
        use inference_model_qwen::config::Qwen3_5Config;
        use inference_model_qwen::loader;
        use inference_backend::config::ModelConfig;
        use burn::backend::ndarray::NdArrayDevice;

        let config_str = std::fs::read_to_string(model_dir.join("config.json")).expect("No config.json");
        let q_cfg: Qwen3_5Config = Qwen3_5Config::from_json(&config_str).expect("Failed to parse config");

        let weights_path = model_dir.join("model.safetensors");
        let weights_data = loader::load_safetensors_data(&weights_path).expect("Failed to load weights");

        let device = NdArrayDevice::default();
        let model = Qwen3_5Model::<NdArray>::new_with_weights(&q_cfg.text_config, &device, &weights_data);

        let model_config = ModelConfig {
            num_layers: q_cfg.text_config.num_hidden_layers,
            num_kv_heads: q_cfg.text_config.num_key_value_heads,
            head_dim: q_cfg.text_config.head_dim,
            vocab_size: q_cfg.text_config.vocab_size,
            is_moe: false,
            num_experts: None,
            top_k_experts: None,
            eos_token_id: q_cfg.text_config.eos_token_id,
        };

        eprintln!(">> Using BurnBackend for Qwen (with loaded weights)");
        Arc::new(BurnBackend::new(model, model_config, device))
    } else {
        Arc::new(CandleBackend::load(&model_dir).expect("Failed to load model"))
    };
    eprintln!(">> Model loaded.");

    let model_config = backend.model_config().clone();
    let num_pages = 512;
    let page_size = 16;

    let engine_config = EngineConfig {
        max_total_tokens: 8192,
        num_kv_pages: num_pages,
        page_size,
        max_prefill_tokens_per_iter: 512,
        max_decode_batch_size: 8,
        model_config,
    };

    let ctx = EngineContext {
        backend: backend.clone(),
        radix_cache: RadixCache::new(page_size, num_pages),
        kv_pool: KvCachePool::new(num_pages, page_size),
        waiting: std::collections::VecDeque::new(),
        prefilling: Vec::new(),
        decoding: Vec::new(),
        config: engine_config,
        stats: EngineStats::default(),
    };

    let (sched_tx, sched_rx) = mpsc::channel(32);
    let (gpu_tx, gpu_rx) = std::sync::mpsc::sync_channel(1);

    tokio::spawn(run_overlapped_loop(ctx, sched_rx, gpu_tx));

    // Run inference on a dedicated thread (candle CPU ops are blocking)
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
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    eprintln!(">> API server listening on http://0.0.0.0:8080");

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c().await.expect("failed to install CTRL+C handler");
            eprintln!(">> Shutting down...");
        })
        .await
        .unwrap();
}
