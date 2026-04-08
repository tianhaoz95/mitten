use axum_test::TestServer;
use inference_api::{create_router, AppState, start_tokenizer_service};
use inference_engine::engine_loop::{run_overlapped_loop, EngineContext, GpuWork};
use inference_engine::config::{EngineConfig, EngineStats};
use inference_engine::kv_pool::KvCachePool;
use inference_engine::radix_cache::RadixCache;
use inference_backend::stub::StubBackend;
use inference_backend::backend::BackendHandle;
use inference_backend::ModelConfig;
use std::sync::Arc;
use tokio::sync::mpsc;
use std::path::Path;
use serde_json::{json, Value};

fn make_model_config() -> ModelConfig {
    ModelConfig {
        num_layers: 2,
        num_kv_heads: 1,
        head_dim: 16,
        vocab_size: 100,
        is_moe: false,
        num_experts: None,
        top_k_experts: None,
        eos_token_id: 0,
    }
}

async fn make_test_server() -> TestServer {
    let model_config = make_model_config();
    let num_pages = 64;
    let page_size = 4;
    let backend = Arc::new(StubBackend::new(model_config.clone(), num_pages, 42));

    let ctx = EngineContext {
        backend: backend.clone(),
        radix_cache: RadixCache::new(page_size, num_pages),
        kv_pool: KvCachePool::new(num_pages, page_size),
        waiting: std::collections::VecDeque::new(),
        prefilling: Vec::new(),
        decoding: Vec::new(),
        config: EngineConfig {
            max_total_tokens: 1024,
            num_kv_pages: num_pages,
            page_size,
            max_prefill_tokens_per_iter: 1024,
            max_decode_batch_size: 32,
            model_config,
        },
        stats: EngineStats::default(),
    };

    let (sched_tx, sched_rx) = mpsc::channel(100);
    let (gpu_tx, gpu_rx) = std::sync::mpsc::sync_channel::<GpuWork>(1);

    tokio::spawn(run_overlapped_loop(ctx, sched_rx, gpu_tx));
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        while let Ok(work) = gpu_rx.recv() {
            let res = rt.block_on(backend.forward(&work.batch));
            let _ = work.result_tx.send(res);
        }
    });

    let tokenizer = start_tokenizer_service(Path::new("test_tokenizer.json")).unwrap();
    let state = AppState { engine_tx: sched_tx, tokenizer, model_name: "test-model".to_string() };
    TestServer::new(create_router(state))
}

#[tokio::test]
async fn test_health_endpoint() {
    let server = make_test_server().await;
    let resp = server.get("/health").await;
    resp.assert_status_ok();
    let body: Value = resp.json();
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn test_models_endpoint() {
    let server = make_test_server().await;
    let resp = server.get("/v1/models").await;
    resp.assert_status_ok();
    let body: Value = resp.json();
    assert!(body["data"].as_array().unwrap().len() >= 1);
    assert_eq!(body["data"][0]["id"], "test-model");
}

#[tokio::test]
async fn test_non_streaming_response_schema() {
    let server = make_test_server().await;
    let resp = server.post("/v1/chat/completions")
        .json(&json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "stream": false
        }))
        .await;
    resp.assert_status_ok();
    let body: Value = resp.json();
    assert!(body["id"].as_str().unwrap().starts_with(""));
    assert_eq!(body["object"], "chat.completion");
    assert!(body["choices"].as_array().unwrap().len() >= 1);
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert!(body["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn test_invalid_role_422() {
    let server = make_test_server().await;
    let resp = server.post("/v1/chat/completions")
        .json(&json!({
            "model": "test-model",
            "messages": [{"role": "unknown_role", "content": "hi"}],
            "max_tokens": 2
        }))
        .await;
    resp.assert_status(axum::http::StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn test_negative_max_tokens_422() {
    let server = make_test_server().await;
    let resp = server.post("/v1/chat/completions")
        .json(&json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": -1
        }))
        .await;
    resp.assert_status(axum::http::StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn test_back_pressure_503() {
    let model_config = make_model_config();
    let num_pages = 64;
    let page_size = 4;
    let backend = Arc::new(StubBackend::new(model_config.clone(), num_pages, 42));

    let ctx = EngineContext {
        backend: backend.clone(),
        radix_cache: RadixCache::new(page_size, num_pages),
        kv_pool: KvCachePool::new(num_pages, page_size),
        waiting: std::collections::VecDeque::new(),
        prefilling: Vec::new(),
        decoding: Vec::new(),
        config: EngineConfig {
            max_total_tokens: 1024,
            num_kv_pages: num_pages,
            page_size,
            max_prefill_tokens_per_iter: 1024,
            max_decode_batch_size: 32,
            model_config,
        },
        stats: EngineStats::default(),
    };

    // Channel with capacity 1; we'll fill it before the test request
    let (sched_tx, sched_rx) = mpsc::channel(1);
    let (gpu_tx, _gpu_rx) = std::sync::mpsc::sync_channel::<GpuWork>(1);
    tokio::spawn(run_overlapped_loop(ctx, sched_rx, gpu_tx));

    let tokenizer = start_tokenizer_service(Path::new("test_tokenizer.json")).unwrap();
    let state = AppState { engine_tx: sched_tx.clone(), tokenizer, model_name: "test-model".to_string() };
    let server = TestServer::new(create_router(state));

    // Fill the channel so try_send fails
    use inference_backend::{Request, SamplingParams};
    use parking_lot::Mutex;
    let dummy = Arc::new(Mutex::new(Request::new(
        uuid::Uuid::new_v4(), vec![1], SamplingParams::default(),
    )));
    sched_tx.try_send(dummy).ok(); // fill the slot

    let resp = server.post("/v1/chat/completions")
        .json(&json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1
        }))
        .await;
    resp.assert_status(axum::http::StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn test_streaming_sse_format() {
    let server = make_test_server().await;
    let resp = server.post("/v1/chat/completions")
        .json(&json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "stream": true
        }))
        .await;
    resp.assert_status_ok();
    // SSE content-type
    let ct = resp.headers().get("content-type").unwrap().to_str().unwrap();
    assert!(ct.contains("text/event-stream"), "Expected SSE content-type, got: {ct}");
}
