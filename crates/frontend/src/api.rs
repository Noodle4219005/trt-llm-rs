use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use trtllm_core::{Millis, RequestId, Result, SamplingParams, TokenId};

/// A request as the engine side sees it, with the arrival instant already
/// stamped by the HTTP layer.
#[derive(Clone, Debug)]
pub struct GenerateRequest {
    pub id: RequestId,
    pub prompt: Vec<TokenId>,
    pub prompt_text: String,
    pub params: SamplingParams,
    pub arrival_ms: Millis,
    pub stream: bool,
}

/// One streamed increment.
#[derive(Clone, Debug)]
pub enum StreamChunk {
    Token { token: TokenId, text: String },
    Done { finish_reason: &'static str },
    Error { message: String },
}

/// What the frontend needs from whatever is behind it: a router, an in-process
/// deployment, or a single worker.
#[async_trait]
pub trait Completions: Send + Sync + 'static {
    async fn generate(&self, req: GenerateRequest) -> Result<mpsc::Receiver<StreamChunk>>;
    fn model_name(&self) -> String;
    /// Encode a prompt. Kept on this trait so the frontend never has to own a
    /// tokenizer that must match the engine's exactly.
    fn encode(&self, text: &str) -> Vec<TokenId>;
    fn decode(&self, tokens: &[TokenId]) -> String;
}

// ---- OpenAI wire types -----------------------------------------------------

#[derive(Clone, Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub ignore_eos: Option<bool>,
}

impl ChatCompletionRequest {
    pub fn flatten_prompt(&self) -> String {
        self.messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn sampling(&self) -> SamplingParams {
        SamplingParams {
            max_tokens: self
                .max_completion_tokens
                .or(self.max_tokens)
                .unwrap_or(200),
            temperature: self.temperature.unwrap_or(0.0),
            top_p: self.top_p.unwrap_or(1.0),
            ignore_eos: self.ignore_eos.unwrap_or(true),
            seed: self.seed,
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub ignore_eos: Option<bool>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: Delta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessageOut,
    pub finish_reason: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChatMessageOut {
    pub role: &'static str,
    pub content: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_completion_tokens_wins_over_the_legacy_field() {
        let r: ChatCompletionRequest =
            serde_json::from_str(r#"{"messages":[],"max_tokens":10,"max_completion_tokens":200}"#)
                .expect("parse");
        assert_eq!(r.sampling().max_tokens, 200);
    }

    #[test]
    fn defaults_match_the_benchmark_shape() {
        let r: ChatCompletionRequest =
            serde_json::from_str(r#"{"messages":[{"role":"user","content":"hi"}]}"#)
                .expect("parse");
        let s = r.sampling();
        assert_eq!(s.max_tokens, 200);
        assert_eq!(s.temperature, 0.0);
        assert!(
            s.ignore_eos,
            "the scored workload pins OSL, so EOS must not cut it short"
        );
        assert_eq!(r.flatten_prompt(), "user: hi");
    }
}
