use inference_api::{create_router, AppState, start_tokenizer_service};
use inference_engine::engine_loop::{run_overlapped_loop, EngineContext};
use inference_engine::config::{EngineConfig, EngineStats};
use inference_engine::kv_pool::KvCachePool;
use inference_engine::radix_cache::RadixCache;
use inference_backend::backend::BackendHandle;
use inference_backend::CandleBackend;
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
    let backend = Arc::new(
        CandleBackend::load(&model_dir).expect("Failed to load model")
    );
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
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        while let Ok(work) = gpu_rx.recv() {
            let res = rt.block_on(BackendHandle::forward(backend.as_ref(), &work.batch));
            let _ = work.result_tx.send(res);
        }
    });

    let tokenizer = start_tokenizer_service(&tokenizer_path).expect("Failed to start tokenizer");

    let state = AppState {
        engine_tx: sched_tx,
        tokenizer,
        model_name: "gemma-4-e2b-it".to_string(),
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
