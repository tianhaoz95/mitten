use crate::engine_loop::EngineContext;
use inference_backend::{Request, RequestState};
use parking_lot::Mutex;
use std::sync::Arc;

pub struct PrefillAdder<'a> {
    ctx: &'a mut EngineContext,
    budget: usize,
    tokens_used: usize,
}

impl<'a> PrefillAdder<'a> {
    pub fn new(ctx: &'a mut EngineContext, budget: usize) -> Self {
        Self {
            ctx,
            budget,
            tokens_used: 0,
        }
    }

    pub fn run(mut self) -> Vec<Arc<Mutex<Request>>> {
        let mut added = Vec::new();

        // 1. Continue in-progress prefill requests.
        // We use a separate list to avoid borrow checker issues while iterating and mutating ctx.prefilling
        let in_progress: Vec<_> = self.ctx.prefilling.clone();
        for req_arc in in_progress {
            if self.tokens_used >= self.budget {
                break;
            }
            if let Some(chunk_size) = self.schedule_prefill_chunk(&req_arc) {
                if chunk_size > 0 {
                    added.push(req_arc);
                }
            }
        }

        // 2. Admit new requests from waiting queue.
        while self.tokens_used < self.budget && !self.ctx.waiting.is_empty() {
            let req_arc = self.ctx.waiting.front().unwrap().clone();
            let input_len = req_arc.lock().input_ids.len();
            let pages_needed = (input_len + self.ctx.kv_pool.page_size - 1) / self.ctx.kv_pool.page_size;

            if self.ctx.kv_pool.free_pages() < pages_needed {
                let deficit = pages_needed - self.ctx.kv_pool.free_pages();
                let freed_pages = self.ctx.radix_cache.evict_pages(deficit);
                for page in freed_pages {
                    self.ctx.kv_pool.free(page);
                }

                if self.ctx.kv_pool.free_pages() < pages_needed {
                    // Cannot admit this request yet.
                    break;
                }
            }

            // Pop it for real
            self.ctx.waiting.pop_front();

            let (cached_len, kv_pages, locked_nodes) = {
                let req = req_arc.lock();
                self.ctx.radix_cache.match_prefix(&req.input_ids)
            };

            {
                let mut req = req_arc.lock();
                req.cached_len = cached_len;
                req.device_len = cached_len;
                req.locked_nodes = locked_nodes;
                req.kv_pages = kv_pages;

                // Allocate remaining pages
                let total_pages_needed = (req.input_ids.len() + self.ctx.kv_pool.page_size - 1) / self.ctx.kv_pool.page_size;
                while req.kv_pages.len() < total_pages_needed {
                    if let Some(page) = self.ctx.kv_pool.allocate(req.id) {
                        req.kv_pages.push(page);
                    } else {
                        // This should not happen if our arithmetic is correct
                        unreachable!("KV pool exhausted despite check");
                    }
                }

                req.state = RequestState::Prefilling {
                    processed_tokens: 0,
                };
            }

            self.ctx.prefilling.push(req_arc.clone());
            if let Some(chunk_size) = self.schedule_prefill_chunk(&req_arc) {
                if chunk_size > 0 {
                    added.push(req_arc);
                }
            }
        }

        added
    }

    fn schedule_prefill_chunk(&mut self, req_arc: &Arc<Mutex<Request>>) -> Option<usize> {
        let mut req = req_arc.lock();
        let remaining = req.input_ids.len() - req.device_len;
        if remaining == 0 {
            return Some(0);
        }

        let chunk = remaining.min(self.budget - self.tokens_used);
        if chunk > 0 {
            req.extend_len = chunk;
            self.tokens_used += chunk;
            Some(chunk)
        } else {
            None
        }
    }
}
