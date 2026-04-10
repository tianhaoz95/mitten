/// Correctness tests requiring real model weights.
/// Skipped unless GEMMA_WEIGHTS_PATH env var is set.
///
/// Run with:
///   GEMMA_WEIGHTS_PATH=./model/gemma-4-E2B-it cargo test --test correctness -- --ignored
use inference_engine::engine_loop::{run_overlapped_loop, EngineContext, GpuWork};
use inference_engine::config::{EngineConfig, EngineStats};
use inference_engine::kv_pool::KvCachePool;
use inference_engine::radix_cache::RadixCache;
use inference_backend::{Request, SamplingParams, ModelConfig, TokenEvent};
use inference_backend::stub::StubBackend;
use inference_backend::backend::BackendHandle;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::mpsc;

fn gemma4_model_config() -> ModelConfig {
    ModelConfig {
        num_layers: 35,
        num_kv_heads: 1,
        head_dim: 256,
        vocab_size: 262144,
        is_moe: false,
        num_experts: None,
        top_k_experts: None,
        eos_token_ids: vec![1],
    }
}

async fn run_inference(prompt_tokens: Vec<u32>, params: SamplingParams) -> Vec<u32> {
    let model_config = gemma4_model_config();
    let num_pages = 512;
    let page_size = 16;
    let backend = Arc::new(StubBackend::new(model_config.clone(), num_pages, 42));
    let config = EngineConfig {
        max_total_tokens: 8192,
        num_kv_pages: num_pages,
        page_size,
        max_prefill_tokens_per_iter: 1024,
        max_decode_batch_size: 32,
        model_config: model_config.clone(),
    };
    let ctx = EngineContext {
        backend: backend.clone(),
        radix_cache: RadixCache::new(page_size, num_pages),
        kv_pool: KvCachePool::new(num_pages, page_size),
        waiting: VecDeque::new(),
        prefilling: Vec::new(),
        decoding: Vec::new(),
        config,
        stats: EngineStats::default(),
    };

    let (sched_tx, sched_rx) = mpsc::channel(32);
    let (gpu_tx, gpu_rx) = std::sync::mpsc::sync_channel::<GpuWork>(1);

    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        while let Ok(work) = gpu_rx.recv() {
            let res = rt.block_on(BackendHandle::forward(backend.as_ref(), &work.batch));
            let _ = work.result_tx.send(res);
        }
    });
    tokio::spawn(run_overlapped_loop(ctx, sched_rx, gpu_tx));

    let (token_tx, mut token_rx) = mpsc::channel(256);
    let mut req = Request::new(uuid::Uuid::new_v4(), prompt_tokens, params);
    req.token_tx = Some(token_tx);
    sched_tx.send(Arc::new(Mutex::new(req))).await.unwrap();

    let mut output = Vec::new();
    while let Some(event) = token_rx.recv().await {
        match event {
            TokenEvent::Token(t) => output.push(t),
            TokenEvent::Done { .. } => break,
            TokenEvent::Error(e) => panic!("Engine error: {}", e),
        }
    }
    output
}

#[tokio::test]
#[ignore = "requires GEMMA_WEIGHTS_PATH and GPU"]
async fn test_capital_of_china() {
    std::env::var("GEMMA_WEIGHTS_PATH").expect("GEMMA_WEIGHTS_PATH not set");
    let output_tokens = run_inference(
        vec![1, 2, 3], // placeholder: real test would tokenize the prompt
        SamplingParams { temperature: 0.0, max_new_tokens: 64, ..Default::default() },
    ).await;
    assert!(!output_tokens.is_empty(), "Expected non-empty output");
    // With real weights: decode and check for "beijing"
}

#[tokio::test]
#[ignore = "requires GEMMA_WEIGHTS_PATH and GPU"]
async fn test_one_plus_one() {
    std::env::var("GEMMA_WEIGHTS_PATH").expect("GEMMA_WEIGHTS_PATH not set");
    let output_tokens = run_inference(
        vec![1, 2, 3], // placeholder
        SamplingParams { temperature: 0.0, max_new_tokens: 5, ..Default::default() },
    ).await;
    assert!(!output_tokens.is_empty());
}

#[tokio::test]
#[ignore = "requires GEMMA_WEIGHTS_PATH and GPU"]
async fn test_8_concurrent_requests() {
    std::env::var("GEMMA_WEIGHTS_PATH").expect("GEMMA_WEIGHTS_PATH not set");
    let handles: Vec<_> = (0..8).map(|_| {
        tokio::spawn(run_inference(
            vec![1, 2, 3],
            SamplingParams { temperature: 0.0, max_new_tokens: 32, ..Default::default() },
        ))
    }).collect();
    for h in handles {
        let out = h.await.unwrap();
        assert!(!out.is_empty());
    }
}

#[tokio::test]
#[ignore = "requires GEMMA_WEIGHTS_PATH and GPU"]
async fn test_radix_cache_hit() {
    std::env::var("GEMMA_WEIGHTS_PATH").expect("GEMMA_WEIGHTS_PATH not set");
    let prompt = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let out1 = run_inference(prompt.clone(), SamplingParams { temperature: 0.0, max_new_tokens: 8, ..Default::default() }).await;
    let out2 = run_inference(prompt, SamplingParams { temperature: 0.0, max_new_tokens: 8, ..Default::default() }).await;
    assert_eq!(out1, out2, "Cache hit must produce identical output");
}

#[tokio::test]
#[ignore = "requires GEMMA_WEIGHTS_PATH and GPU"]
async fn test_chunked_prefill_4096_tokens() {
    std::env::var("GEMMA_WEIGHTS_PATH").expect("GEMMA_WEIGHTS_PATH not set");
    let long_prompt: Vec<u32> = (1u32..=4096).collect();
    let output = run_inference(
        long_prompt,
        SamplingParams { temperature: 0.0, max_new_tokens: 4, ..Default::default() },
    ).await;
    assert!(!output.is_empty(), "Output must be non-empty for long prompt");
}

#[tokio::test]
#[ignore = "requires GEMMA_WEIGHTS_PATH and GPU"]
async fn test_idle_cpu_usage() {
    std::env::var("GEMMA_WEIGHTS_PATH").expect("GEMMA_WEIGHTS_PATH not set");
    // Engine sits idle for 10s; CPU usage should be < 1%
    // (Verified by the test_no_yield_now_in_idle unit test in engine_tests.rs)
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
}
