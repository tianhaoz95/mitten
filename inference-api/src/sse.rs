use crate::types::{StreamChunk, StreamChoice, DeltaMessage};
use crate::TokenizerHandle;
use inference_backend::{TokenEvent, FinishReason};
use tokio::sync::mpsc;
use futures::stream::Stream;
use std::pin::Pin;
use axum::response::sse::Event;

pub fn build_sse_stream(
    mut token_rx: mpsc::Receiver<TokenEvent>,
    tokenizer: TokenizerHandle,
    model: String,
    id: String,
) -> Pin<Box<dyn Stream<Item = Result<Event, std::convert::Infallible>> + Send>> {
    let stream = async_stream::stream! {
        while let Some(event) = token_rx.recv().await {
            match event {
                TokenEvent::Token(token_id) => {
                    if let Ok(text) = tokenizer.decode(vec![token_id]).await {
                        let chunk = StreamChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
                            model: model.clone(),
                            choices: vec![StreamChoice {
                                index: 0,
                                delta: DeltaMessage {
                                    role: None,
                                    content: Some(text),
                                },
                                finish_reason: None,
                            }],
                        };
                        if let Ok(json) = serde_json::to_string(&chunk) {
                            yield Ok(Event::default().data(json));
                        }
                    }
                }
                TokenEvent::Done { finish_reason } => {
                    let reason_str = match finish_reason {
                        FinishReason::EosToken => "stop",
                        FinishReason::MaxTokens => "length",
                        _ => "stop",
                    };
                    let chunk = StreamChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
                        model: model.clone(),
                        choices: vec![StreamChoice {
                            index: 0,
                            delta: DeltaMessage::default(),
                            finish_reason: Some(reason_str.to_string()),
                        }],
                    };
                    if let Ok(json) = serde_json::to_string(&chunk) {
                        yield Ok(Event::default().data(json));
                    }
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                TokenEvent::Error(e) => {
                    yield Ok(Event::default().data(format!("Error: {}", e)));
                    break;
                }
            }
        }
    };
    Box::pin(stream)
}
