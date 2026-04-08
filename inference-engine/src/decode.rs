use crate::engine_loop::EngineContext;
use inference_backend::{Request, RequestState};
use parking_lot::Mutex;
use std::sync::Arc;

pub fn collect_decode_requests(ctx: &mut EngineContext) -> Vec<Arc<Mutex<Request>>> {
    let mut decode_reqs = Vec::new();

    // Iterate through decoding list and keep only those still in Decoding state
    let mut i = 0;
    while i < ctx.decoding.len() {
        let req_arc = ctx.decoding[i].clone();
        let mut req = req_arc.lock();

        match req.state {
            RequestState::Decoding => {
                decode_reqs.push(req_arc.clone());
                i += 1;
            }
            RequestState::Done | RequestState::Aborted { .. } => {
                // Release resources
                ctx.radix_cache.unlock_nodes(&req.locked_nodes);
                req.locked_nodes.clear();

                for &page in &req.kv_pages {
                    ctx.kv_pool.free(page);
                }
                req.kv_pages.clear();

                // Remove from decoding list
                ctx.decoding.remove(i);
                // Don't increment i
            }
            _ => {
                // Should not happen, but we'll just keep it
                i += 1;
            }
        }
    }

    // Enforce max decode batch size
    if decode_reqs.len() > ctx.config.max_decode_batch_size {
        decode_reqs.truncate(ctx.config.max_decode_batch_size);
    }

    decode_reqs
}
