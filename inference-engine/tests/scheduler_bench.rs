use inference_engine::engine_loop::EngineContext;
use inference_engine::config::{EngineConfig, EngineStats};
use inference_engine::kv_pool::KvCachePool;
use inference_engine::radix_cache::RadixCache;
use inference_engine::scheduler::schedule_batch;
use inference_backend::{Request, SamplingParams, ModelConfig, RequestState};
use inference_backend::stub::StubBackend;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;

fn make_ctx(num_pages: usize, page_size: usize) -> EngineContext {
    let model_config = ModelConfig {
        num_layers: 2,
        num_kv_heads: 1,
        head_dim: 16,
        vocab_size: 100,
        is_moe: false,
        num_experts: None,
        top_k_experts: None,
        eos_token_ids: vec![99],
    };
    let backend = Arc::new(StubBackend::new(model_config.clone(), num_pages, 42));
    let config = EngineConfig {
        max_total_tokens: 65536,
        num_kv_pages: num_pages,
        page_size,
        max_prefill_tokens_per_iter: 1024,
        max_decode_batch_size: 64,
        model_config,
    };
    EngineContext {
        backend,
        radix_cache: RadixCache::new(page_size, num_pages),
        kv_pool: KvCachePool::new(num_pages, page_size),
        waiting: VecDeque::new(),
        prefilling: Vec::new(),
        decoding: Vec::new(),
        config,
        stats: EngineStats::default(),
    }
}

/// Scheduler overhead test: 1000 scheduling iterations with 32 decode requests.
/// Total scheduling time (excluding forward) must be < 1000ms (< 1ms/iter).
#[test]
fn test_scheduler_overhead() {
    let mut ctx = make_ctx(4096, 16);

    // Pre-populate 32 decoding requests
    for i in 0..32u32 {
        let mut req = Request::new(
            uuid::Uuid::new_v4(),
            vec![i; 16],
            SamplingParams { max_new_tokens: 2000, ..Default::default() },
        );
        req.state = RequestState::Decoding;
        req.device_len = 16;
        req.output_ids = vec![i];
        // Allocate pages
        for _ in 0..1 {
            if let Some(page) = ctx.kv_pool.allocate(req.id) {
                req.kv_pages.push(page);
            }
        }
        ctx.decoding.push(Arc::new(Mutex::new(req)));
    }

    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _batch = schedule_batch(&mut ctx);
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 1000,
        "Scheduler overhead too high: {}ms for 1000 iterations (limit: 1000ms)",
        elapsed.as_millis()
    );
}
