use crate::types::ChatMessage;
use std::path::Path;
use tokenizers::Tokenizer;
use tokio::sync::{mpsc, oneshot};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("tokenizer error: {0}")]
    Internal(String),
}

#[derive(Clone)]
pub struct TokenizerHandle {
    encode_tx: mpsc::Sender<EncodeRequest>,
    decode_tx: mpsc::Sender<DecodeRequest>,
}

struct EncodeRequest {
    messages: Vec<ChatMessage>,
    result_tx: oneshot::Sender<Result<Vec<u32>, TokenizerError>>,
}

struct DecodeRequest {
    token_ids: Vec<u32>,
    result_tx: oneshot::Sender<Result<String, TokenizerError>>,
}

impl TokenizerHandle {
    pub async fn encode(&self, messages: Vec<ChatMessage>) -> Result<Vec<u32>, TokenizerError> {
        let (tx, rx) = oneshot::channel();
        self.encode_tx.send(EncodeRequest {
            messages,
            result_tx: tx,
        }).await.map_err(|_| TokenizerError::Internal("Tokenizer task closed".to_string()))?;
        rx.await.map_err(|_| TokenizerError::Internal("Tokenizer result channel closed".to_string()))?
    }

    pub async fn decode(&self, token_ids: Vec<u32>) -> Result<String, TokenizerError> {
        let (tx, rx) = oneshot::channel();
        self.decode_tx.send(DecodeRequest {
            token_ids,
            result_tx: tx,
        }).await.map_err(|_| TokenizerError::Internal("Tokenizer task closed".to_string()))?;
        rx.await.map_err(|_| TokenizerError::Internal("Tokenizer result channel closed".to_string()))?
    }
}

pub fn start_tokenizer_service(tokenizer_path: &Path) -> Result<TokenizerHandle, TokenizerError> {
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| TokenizerError::Internal(e.to_string()))?;
    
    let (encode_tx, mut encode_rx) = mpsc::channel::<EncodeRequest>(32);
    let (decode_tx, mut decode_rx) = mpsc::channel::<DecodeRequest>(32);

    tokio::spawn(async move {
        loop {
            tokio::select! {
                Some(req) = encode_rx.recv() => {
                    // Gemma 4 chat template: <bos><|turn>role\ncontent<turn|>\n...<|turn>model\n
                    let mut text = String::from("<bos>");
                    for msg in &req.messages {
                        let role = match msg.role {
                            crate::types::Role::System => "system",
                            crate::types::Role::User => "user",
                            crate::types::Role::Assistant => "model",
                        };
                        text.push_str(&format!("<|turn>{}\n{}<turn|>\n", role, msg.content));
                    }
                    // REMOVED trailing \n after model header
                    text.push_str("<|turn>model");
                    eprintln!(">> Encoding text: {:?}", text);
                    let res = tokenizer.encode(text, false)
                        .map(|enc| enc.get_ids().to_vec())
                        .map_err(|e| TokenizerError::Internal(e.to_string()));
                    let _ = req.result_tx.send(res);
                }
                Some(req) = decode_rx.recv() => {
                    let res = tokenizer.decode(&req.token_ids, true)
                        .map_err(|e| TokenizerError::Internal(e.to_string()));
                    let _ = req.result_tx.send(res);
                }
                else => break,
            }
        }
    });

    Ok(TokenizerHandle {
        encode_tx,
        decode_tx,
    })
}
