use crate::config::{EngineConfig, EngineStats};
use crate::kv_pool::KvCachePool;
use crate::radix_cache::RadixCache;
use inference_backend::backend::BackendHandle;
use inference_backend::{Batch, Request, RequestState, AbortReason, TokenEvent};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

pub struct EngineContext {
    pub backend: Arc<dyn BackendHandle>,
    pub radix_cache: RadixCache,
    pub kv_pool: KvCachePool,
    pub waiting: VecDeque<Arc<Mutex<Request>>>,
    pub prefilling: Vec<Arc<Mutex<Request>>>,
    pub decoding: Vec<Arc<Mutex<Request>>>,
    pub config: EngineConfig,
    pub stats: EngineStats,
    pub detokenizer: Option<Arc<dyn Fn(&[u32]) -> String + Send + Sync>>,
}

pub struct GpuWork {
    pub batch: Batch,
    pub result_tx: oneshot::Sender<Result<inference_backend::backend::Logits, inference_backend::EngineError>>,
}

pub async fn run_overlapped_loop(
    mut ctx: EngineContext,
    mut sched_rx: mpsc::Receiver<Arc<Mutex<Request>>>,
    gpu_tx: std::sync::mpsc::SyncSender<GpuWork>,
) {
    let mut in_flight: Option<(Batch, oneshot::Receiver<Result<inference_backend::backend::Logits, inference_backend::EngineError>>)> = None;

    loop {
        // 1. Drain incoming requests
        while let Ok(req) = sched_rx.try_recv() {
            ctx.waiting.push_back(req);
        }

        // 2. Collect results from previous forward pass
        if let Some((prev_batch, result_rx)) = in_flight.take() {
            match result_rx.await {
                Ok(Ok(logits)) => {
                    crate::scheduler::process_and_transition(&mut ctx, &prev_batch, logits);
                }
                Ok(Err(e)) => {
                    // Abort all requests in the batch
                    for req_arc in &prev_batch.requests {
                        let mut req = req_arc.lock();
                        req.state = RequestState::Aborted {
                            reason: AbortReason::BackendError(e.to_string()),
                        };
                        if let Some(tx) = &req.token_tx {
                            let _ = tx.try_send(TokenEvent::Error(e.to_string()));
                        }
                    }
                }
                Err(_) => {
                    panic!("GPU thread channel closed");
                }
            }
        }

        // 3. Schedule the next batch
        let next_batch = crate::scheduler::schedule_batch(&mut ctx);

        if let Some(batch) = next_batch {
            // 4. Fire prefetch (no-op in Phase 0)
            ctx.backend.prefetch(&batch);

            // 5. Send to GPU thread
            let (result_tx, result_rx) = oneshot::channel();
            if let Err(_) = gpu_tx.send(GpuWork {
                batch: batch.clone(),
                result_tx,
            }) {
                panic!("Failed to send work to GPU thread");
            }
            in_flight = Some((batch, result_rx));
        } else {
            // Nothing schedulable. Check if GPU is still draining or truly idle.
            if let Some((prev_batch, result_rx)) = in_flight.take() {
                // GPU draining: wait for it
                match result_rx.await {
                    Ok(Ok(logits)) => {
                        crate::scheduler::process_and_transition(&mut ctx, &prev_batch, logits);
                    }
                    _ => {}
                }
                // Check for new requests again before continuing
                while let Ok(req) = sched_rx.try_recv() {
                    ctx.waiting.push_back(req);
                }
            } else {
                // Truly idle: wait for a new request
                if let Some(req) = sched_rx.recv().await {
                    ctx.waiting.push_back(req);
                    // Drain others
                    while let Ok(req) = sched_rx.try_recv() {
                        ctx.waiting.push_back(req);
                    }
                }
            }
        }
    }
}
