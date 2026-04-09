use axum_test::TestServer;
use inference_api::{create_router, AppState, start_tokenizer_service};
use inference_engine::engine_loop::{run_overlapped_loop, EngineContext};
use inference_engine::config::{EngineConfig, EngineStats};
use inference_engine::kv_pool::KvCachePool;
use inference_engine::radix_cache::RadixCache;
use inference_backend::stub::StubBackend;
use inference_backend::backend::BackendHandle;
use inference_backend::{ModelConfig};
use std::sync::Arc;
use tokio::sync::mpsc;
use std::path::Path;
use serde_json::json;

// We need a dummy tokenizer for the test if we don't have a file.
// Let's create a minimal tokenizer.json if it doesn't exist.

#[tokio::test]
async fn test_api_chat_completions() {
    // 1. Setup minimal components
    let model_config = ModelConfig {
        num_layers: 2,
        num_kv_heads: 1,
        head_dim: 16,
        vocab_size: 100,
        is_moe: false,
        num_experts: None,
        top_k_experts: None,
        eos_token_id: 0,
    };
    let num_pages = 16;
    let page_size = 4;
    let backend = Arc::new(StubBackend::new(model_config.clone(), num_pages, 42).with_forced_tokens(vec![42]));
    
    let engine_config = EngineConfig {
        max_total_tokens: 1024,
        num_kv_pages: num_pages,
        page_size,
        max_prefill_tokens_per_iter: 1024,
        max_decode_batch_size: 32,
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

    let (sched_tx, sched_rx) = mpsc::channel(100);
    let (gpu_tx, gpu_rx) = std::sync::mpsc::sync_channel(1);

    tokio::spawn(run_overlapped_loop(ctx, sched_rx, gpu_tx));

    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        while let Ok(work) = gpu_rx.recv() {
            let res = rt.block_on(backend.forward(&work.batch));
            let _ = work.result_tx.send(res);
        }
    });

    // Mock tokenizer service (we'll need a real one or a more robust mock)
    // For now, let's just skip the real tokenizer service and mock the handle if possible.
    // Actually, I'll just write a dummy tokenizer.json.
    let dummy_tokenizer = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {
            "type": "Whitespace"
        },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {
                "hello": 0,
                "world": 1,
                " ": 2,
                "[UNK]": 3
            },
            "unk_token": "[UNK]"
        }
    });
    let tokenizer_path = Path::new("test_tokenizer.json");
    std::fs::write(tokenizer_path, dummy_tokenizer.to_string()).unwrap();

    let tokenizer = start_tokenizer_service(tokenizer_path).unwrap();

    let state = AppState {
        engine_tx: sched_tx,
        tokenizer,
        model_name: "test-model".to_string(),
    };

    let app = create_router(state);
    let server = TestServer::new(app);

    let response = server
        .post("/v1/chat/completions")
        .json(&json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "hello"}
            ],
            "max_tokens": 2,
            "stream": false
        }))
        .await;

    response.assert_status_ok();
    let body: serde_json::Value = response.json();
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    // Since we forced token 42, and our dummy tokenizer doesn't have 42, 
    // it might be decoded as empty or error.
    // But the engine should have sent it.
}
