use crate::decode::collect_decode_requests;
use crate::engine_loop::EngineContext;
use crate::prefill::PrefillAdder;
use inference_backend::{Batch, BatchPhase, Logits, Request, RequestState, TokenEvent, FinishReason};
use parking_lot::Mutex;
use std::sync::Arc;

pub fn schedule_batch(ctx: &mut EngineContext) -> Option<Batch> {
    // 1. Collect decode requests
    let decode_reqs = collect_decode_requests(ctx);

    // 2. Add prefill requests up to token budget
    let decode_tokens = decode_reqs.len();
    let budget = ctx
        .config
        .max_prefill_tokens_per_iter
        .saturating_sub(decode_tokens);
    
    let prefill_reqs = PrefillAdder::new(ctx, budget).run();

    if decode_reqs.is_empty() && prefill_reqs.is_empty() {
        return None;
    }

    // 3. Assemble batch
    Some(assemble_batch(ctx, prefill_reqs, decode_reqs))
}

pub fn assemble_batch(
    _ctx: &EngineContext,
    prefill_reqs: Vec<Arc<Mutex<Request>>>,
    decode_reqs: Vec<Arc<Mutex<Request>>>,
) -> Batch {
    let mut input_ids = Vec::new();
    let mut position_ids = Vec::new();
    let mut page_table = Vec::new();
    let mut all_reqs = Vec::new();

    // Prefill requests first
    for req_arc in &prefill_reqs {
        let req = req_arc.lock();
        let start = req.device_len;
        let end = req.device_len + req.extend_len;
        input_ids.extend_from_slice(&req.input_ids[start..end]);
        for i in start..end {
            position_ids.push(i as u32);
        }
        page_table.push(req.kv_pages.clone());
        all_reqs.push(req_arc.clone());
    }

    // Decode requests
    for req_arc in &decode_reqs {
        let req = req_arc.lock();
        let last_token = *req.output_ids.last().unwrap_or(req.input_ids.last().unwrap());
        input_ids.push(last_token);
        position_ids.push(req.device_len as u32);
        page_table.push(req.kv_pages.clone());
        all_reqs.push(req_arc.clone());
    }

    let phase = match (prefill_reqs.is_empty(), decode_reqs.is_empty()) {
        (false, true) => BatchPhase::PrefillOnly,
        (true, false) => BatchPhase::DecodeOnly,
        _ => BatchPhase::Mixed,
    };

    Batch {
        phase,
        requests: all_reqs,
        input_ids,
        position_ids,
        page_table,
        num_kv_slots: 0, // Filled if needed by backend
        expert_routing: None,
    }
}

pub fn process_and_transition(ctx: &mut EngineContext, batch: &Batch, logits: Logits) {
    let mut row_offset = 0;
    
    // Logits contains one row per request for prefill (last token), and one row per request for decode.
    // Wait, the spec says:
    // "For prefill batches: one row per request (last-position logits only)."
    // "For decode batches: one row per request."
    // "For mixed batches: prefill requests first, then decode requests, in batch order."

    for req_arc in &batch.requests {
        let mut req = req_arc.lock();
        
        // Find if it was prefill or decode in this batch
        // In assemble_batch, we put all prefill first, then all decode.
        
        let is_prefill = matches!(req.state, RequestState::Prefilling { .. });
        
        if is_prefill {
            req.device_len += req.extend_len;
            req.extend_len = 0;
            
            if req.device_len == req.input_ids.len() {
                // Transition to decoding
                let row = &logits.data[row_offset * logits.vocab_size .. (row_offset + 1) * logits.vocab_size];
                let sampled_token = sample_token(row, &req.params);
                
                req.output_ids.push(sampled_token);
                req.state = RequestState::Decoding;
                
                // Move from prefilling to decoding list in ctx
                if let Some(pos) = ctx.prefilling.iter().position(|r| Arc::ptr_eq(r, req_arc)) {
                    ctx.prefilling.remove(pos);
                    ctx.decoding.push(req_arc.clone());
                }
                
                // Promote to radix cache
                ctx.radix_cache.promote_request(&req);
                
                // Send token event
                if let Some(tx) = &req.token_tx {
                    let _ = tx.try_send(TokenEvent::Token(sampled_token));
                }
                
                row_offset += 1;
            } else {
                // Still prefilling (chunked prefill), no logits produced yet for this request if it's not the last chunk?
                // Actually, the backend might only produce logits for the last token of the chunk.
                // But the engine only cares about the very last token of the whole prompt.
                // Let's assume the backend only returns logits for the requests that finished their prefill in this batch.
                // Wait, if it's chunked prefill, do we get logits for each chunk?
                // Usually no, unless we need to sample.
                // Let's re-read the spec for Logits.
                // "For prefill batches: one row per request (last-position logits only)."
                // This implies even for chunked prefill, it might produce logits.
                // But we only sample when device_len == input_ids.len().
                
                // If it's not finished, we just increment row_offset if the backend produced a row for it.
                // Let's assume the backend produces one row per prefill request in the batch.
                row_offset += 1;
            }
        } else if matches!(req.state, RequestState::Decoding) {
            let row = &logits.data[row_offset * logits.vocab_size .. (row_offset + 1) * logits.vocab_size];
            let sampled_token = sample_token(row, &req.params);
            
            req.output_ids.push(sampled_token);
            req.device_len += 1;
            
            // Check for stop conditions
            let mut done = false;
            let mut finish_reason = FinishReason::EosToken; // Default
            
            if sampled_token == ctx.config.model_config.eos_token_id {
                done = true;
                finish_reason = FinishReason::EosToken;
            } else if req.output_ids.len() >= req.params.max_new_tokens {
                done = true;
                finish_reason = FinishReason::MaxTokens;
            }
            // TODO: Stop strings
            
            if let Some(tx) = &req.token_tx {
                let _ = tx.try_send(TokenEvent::Token(sampled_token));
            }

            if done {
                req.state = RequestState::Done;
                // Free KV pages and unlock radix nodes immediately
                ctx.radix_cache.unlock_nodes(&req.locked_nodes);
                req.locked_nodes.clear();
                for &page in &req.kv_pages {
                    ctx.kv_pool.free(page);
                }
                req.kv_pages.clear();
                if let Some(pos) = ctx.decoding.iter().position(|r| Arc::ptr_eq(r, req_arc)) {
                    ctx.decoding.remove(pos);
                }
                if let Some(tx) = &req.token_tx {
                    let _ = tx.try_send(TokenEvent::Done { finish_reason });
                }
            }
            
            row_offset += 1;
        }
    }
}

fn sample_token(logits: &[f32], params: &inference_backend::SamplingParams) -> u32 {
    let temperature = params.temperature;
    if temperature <= 0.0 {
        // Greedy
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0;
        for (i, &val) in logits.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        return max_idx as u32;
    }

    // Temperature sampling (simplified)
    // We don't have rand, so we use a simple hash of logits as entropy
    let mut hash = 0u64;
    for &l in logits.iter().take(100) {
        hash = hash.wrapping_mul(31).wrapping_add(l.to_bits() as u64);
    }
    let rand_val = (hash % 1000) as f32 / 1000.0;

    // Softmax with temperature
    let mut exp_logits: Vec<f32> = logits.iter().map(|&l| (l / temperature as f32).exp()).collect();
    let sum: f32 = exp_logits.iter().sum();
    for l in exp_logits.iter_mut() { *l /= sum; }

    // Cumulative sum
    let mut cumsum = 0.0f32;
    for (i, &p) in exp_logits.iter().enumerate() {
        cumsum += p;
        if rand_val <= cumsum {
            return i as u32;
        }
    }
    (logits.len() - 1) as u32
}
