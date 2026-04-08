use crate::types::{ChatCompletionRequest, ChatCompletionResponse, Choice, ChatMessage, Role, UsageStats};
use crate::tokenizer_service::TokenizerHandle;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Sse},
    routing::{get, post},
    Json, Router,
};
use inference_backend::{Request, SamplingParams, TokenEvent, FinishReason};
use parking_lot::Mutex;
use std::sync::Arc;
use tokio::sync::mpsc;
use uuid::Uuid;

#[derive(Clone)]
pub struct AppState {
    pub engine_tx: mpsc::Sender<Arc<Mutex<Request>>>,
    pub tokenizer: TokenizerHandle,
    pub model_name: String,
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(health_check))
        .with_state(state)
}

async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": state.model_name,
                "object": "model",
                "created": 1686935002,
                "owned_by": "mitten"
            }
        ]
    }))
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    // 422 validation
    if let Some(max_tokens) = req.max_tokens {
        if max_tokens <= 0 {
            return (StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::json!({"error": "max_tokens must be positive"}))).into_response();
        }
    }
    // Validate roles (Role enum handles unknown via serde, but check explicitly)
    for msg in &req.messages {
        match msg.role {
            Role::System | Role::User | Role::Assistant => {}
        }
    }

    // 503 back-pressure check before doing any work
    if state.engine_tx.capacity() == 0 {
        return (StatusCode::SERVICE_UNAVAILABLE, "Engine queue full").into_response();
    }

    // 1. Encode prompt
    let input_ids = match state.tokenizer.encode(req.messages.clone()).await {
        Ok(ids) => ids,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };

    // 2. Create request
    let id = Uuid::new_v4();
    let mut params = SamplingParams::default();
    if let Some(t) = req.temperature { params.temperature = t; }
    if let Some(p) = req.top_p { params.top_p = p; }
    if let Some(k) = req.top_k { params.top_k = k; }
    if let Some(m) = req.max_tokens { params.max_new_tokens = m as usize; }
    params.stream = req.stream;

    let (token_tx, mut token_rx) = mpsc::channel(100);
    let mut request = Request::new(id, input_ids, params);
    request.token_tx = Some(token_tx);

    let req_arc = Arc::new(Mutex::new(request));

    // 3. Send to engine (503 on back-pressure)
    if state.engine_tx.try_send(req_arc.clone()).is_err() {
        return (StatusCode::SERVICE_UNAVAILABLE, "Engine queue full").into_response();
    }

    if req.stream {
        let stream = crate::sse::build_sse_stream(token_rx, state.tokenizer.clone(), state.model_name.clone(), id.to_string());
        Sse::new(stream).into_response()
    } else {
        let mut completion_tokens = 0;
        let mut full_content = String::new();
        let mut finish_reason = "stop".to_string();

        while let Some(event) = token_rx.recv().await {
            match event {
                TokenEvent::Token(id) => {
                    completion_tokens += 1;
                    if let Ok(text) = state.tokenizer.decode(vec![id]).await {
                        full_content.push_str(&text);
                    }
                }
                TokenEvent::Done { finish_reason: reason } => {
                    finish_reason = match reason {
                        FinishReason::MaxTokens => "length".to_string(),
                        _ => "stop".to_string(),
                    };
                    break;
                }
                TokenEvent::Error(e) => {
                    return (StatusCode::INTERNAL_SERVER_ERROR, e).into_response();
                }
            }
        }

        let prompt_tokens = req_arc.lock().input_ids.len();
        let response = ChatCompletionResponse {
            id: id.to_string(),
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            model: state.model_name,
            choices: vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: Role::Assistant,
                    content: full_content,
                },
                finish_reason: Some(finish_reason),
            }],
            usage: UsageStats {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        };
        Json(response).into_response()
    }
}
