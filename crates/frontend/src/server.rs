use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream::Stream;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use crate::api::*;
use trtllm_core::{Millis, RequestId};

pub struct AppState {
    pub backend: Arc<dyn Completions>,
    pub started: Instant,
    next_id: AtomicU64,
}

impl AppState {
    pub fn new(backend: Arc<dyn Completions>) -> Self {
        Self {
            backend,
            started: Instant::now(),
            next_id: AtomicU64::new(0),
        }
    }

    fn next_id(&self) -> RequestId {
        RequestId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Milliseconds since the server started. A monotonic clock, deliberately:
    /// every deadline in this system is relative, and a wall clock that steps
    /// would move a deadline underneath a request in flight.
    fn now_ms(&self) -> Millis {
        self.started.elapsed().as_secs_f64() * 1000.0
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .with_state(state)
}

pub async fn serve(state: Arc<AppState>, addr: std::net::SocketAddr) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "frontend listening");
    axum::serve(listener, router(state)).await
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn models(State(s): State<Arc<AppState>>) -> impl IntoResponse {
    let name = s.backend.model_name();
    Json(serde_json::json!({
        "object": "list",
        "data": [{ "id": name, "object": "model", "owned_by": "trt-llm-rs" }]
    }))
}

fn bad_request(msg: String) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({ "error": { "message": msg } })),
    )
        .into_response()
}

async fn chat_completions(
    State(s): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    if req.messages.is_empty() {
        return bad_request("messages must not be empty".into());
    }
    let text = req.flatten_prompt();
    let stream = req.stream.unwrap_or(false);
    let gen = GenerateRequest {
        id: s.next_id(),
        prompt: s.backend.encode(&text),
        prompt_text: text,
        params: req.sampling(),
        arrival_ms: s.now_ms(),
        stream,
    };
    dispatch(s, gen, req.model, stream).await
}

async fn completions(
    State(s): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let stream = req.stream.unwrap_or(false);
    let gen = GenerateRequest {
        id: s.next_id(),
        prompt: s.backend.encode(&req.prompt),
        prompt_text: req.prompt.clone(),
        params: trtllm_core::SamplingParams {
            max_tokens: req.max_tokens.unwrap_or(200),
            temperature: req.temperature.unwrap_or(0.0),
            ignore_eos: req.ignore_eos.unwrap_or(true),
            ..Default::default()
        },
        arrival_ms: s.now_ms(),
        stream,
    };
    dispatch(s, gen, req.model, stream).await
}

async fn dispatch(
    s: Arc<AppState>,
    gen: GenerateRequest,
    model: Option<String>,
    stream: bool,
) -> Response {
    let id = gen.id;
    let prompt_tokens = gen.prompt.len() as u32;
    let model = model.unwrap_or_else(|| s.backend.model_name());

    let rx = match s.backend.generate(gen).await {
        Ok(rx) => rx,
        Err(e) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({ "error": { "message": e.to_string() } })),
            )
                .into_response()
        }
    };

    if stream {
        sse_response(id, model, rx).into_response()
    } else {
        collect_response(id, model, prompt_tokens, rx).await
    }
}

fn sse_response(
    id: RequestId,
    model: String,
    rx: tokio::sync::mpsc::Receiver<StreamChunk>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let created = unix_now();
    let stream = ReceiverStream::new(rx).map(move |chunk| {
        let payload = match chunk {
            StreamChunk::Token { text, .. } => {
                let c = ChatCompletionChunk {
                    id: id.to_string(),
                    object: "chat.completion.chunk",
                    created,
                    model: model.clone(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: Some(text),
                        },
                        finish_reason: None,
                    }],
                };
                serde_json::to_string(&c).unwrap_or_default()
            }
            StreamChunk::Done { finish_reason } => {
                let c = ChatCompletionChunk {
                    id: id.to_string(),
                    object: "chat.completion.chunk",
                    created,
                    model: model.clone(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: None,
                        },
                        finish_reason: Some(finish_reason),
                    }],
                };
                serde_json::to_string(&c).unwrap_or_default()
            }
            StreamChunk::Error { message } => {
                serde_json::json!({ "error": { "message": message } }).to_string()
            }
        };
        Ok(Event::default().data(payload))
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

async fn collect_response(
    id: RequestId,
    model: String,
    prompt_tokens: u32,
    mut rx: tokio::sync::mpsc::Receiver<StreamChunk>,
) -> Response {
    let mut text = String::new();
    let mut completion_tokens = 0u32;
    let mut finish_reason = "length";
    while let Some(chunk) = rx.recv().await {
        match chunk {
            StreamChunk::Token { text: t, .. } => {
                text.push_str(&t);
                completion_tokens += 1;
            }
            StreamChunk::Done { finish_reason: r } => {
                finish_reason = r;
                break;
            }
            StreamChunk::Error { message } => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": { "message": message } })),
                )
                    .into_response()
            }
        }
    }
    Json(ChatCompletionResponse {
        id: id.to_string(),
        object: "chat.completion",
        created: unix_now(),
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageOut {
                role: "assistant",
                content: text,
            },
            finish_reason,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
    .into_response()
}
