use async_trait::async_trait;
use serde::Serialize;
use thiserror::Error;
use trtllm_core::{Millis, Request, RequestId, SamplingParams, TokenId};

use crate::telemetry::RequestTelemetry;

/// The request payload expected by a Python TensorRT-LLM worker.
#[derive(Clone, Debug, Serialize)]
pub struct TransportRequest {
    pub request_id: String,
    pub prompt_token_ids: Vec<TokenId>,
    pub sampling: SamplingParams,
}

impl TransportRequest {
    pub fn from_request(request: &Request) -> Self {
        Self {
            request_id: request.id.to_string(),
            prompt_token_ids: request.prompt.clone(),
            sampling: request.params.clone(),
        }
    }
}

/// A raw server-sent event. Production HTTP/SSE code belongs behind
/// [`Transport`]; this core only understands the stable payload contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

impl SseEvent {
    pub fn data(data: impl Into<String>) -> Self {
        Self {
            event: None,
            data: data.into(),
        }
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("transport error: {message}")]
pub struct TransportError {
    message: String,
}

impl TransportError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// The only production I/O seam needed by the core. Implementations may use
/// HTTP/SSE, Unix sockets, or an in-process Python bridge.
#[async_trait]
pub trait Transport: Send + 'static {
    async fn send(&mut self, request: TransportRequest) -> Result<(), TransportError>;
    async fn next_event(&mut self) -> Result<Option<SseEvent>, TransportError>;

    /// Cancels locally by dropping the response stream. No imaginary remote
    /// cancel endpoint is part of this worker protocol.
    fn cancel(&mut self, request_id: RequestId) -> Result<(), TransportError>;
}

/// Creates one owned transport per request. Implementations may keep shared,
/// synchronized connection-pool state in the factory; the returned transport
/// itself does not need to implement `Clone`.
pub trait TransportFactory: Send + Sync + 'static {
    type Transport: Transport;

    fn open(&self) -> Result<Self::Transport, TransportError>;
}

/// A concrete `POST <base_url>/generate` HTTP/SSE factory for the Python
/// TensorRT-LLM worker. It is feature-gated with the Dynamo integration so the
/// transport-neutral core remains usable without an HTTP client.
#[cfg(feature = "http-transport")]
#[derive(Clone, Debug)]
pub struct HttpTransportFactory {
    client: reqwest::Client,
    generate_url: String,
}

#[cfg(feature = "http-transport")]
impl HttpTransportFactory {
    pub fn new(base_url: impl AsRef<str>) -> Self {
        let base_url = base_url.as_ref().trim_end_matches('/');
        Self {
            client: reqwest::Client::new(),
            generate_url: format!("{base_url}/generate"),
        }
    }
}

#[cfg(feature = "http-transport")]
impl TransportFactory for HttpTransportFactory {
    type Transport = HttpTransport;

    fn open(&self) -> Result<Self::Transport, TransportError> {
        Ok(HttpTransport {
            client: self.client.clone(),
            generate_url: self.generate_url.clone(),
            stream: None,
            buffered: Vec::new(),
        })
    }
}

#[cfg(feature = "http-transport")]
pub struct HttpTransport {
    client: reqwest::Client,
    generate_url: String,
    stream: Option<futures::stream::BoxStream<'static, Result<bytes::Bytes, reqwest::Error>>>,
    buffered: Vec<u8>,
}

#[cfg(feature = "http-transport")]
#[async_trait]
impl Transport for HttpTransport {
    async fn send(&mut self, request: TransportRequest) -> Result<(), TransportError> {
        use futures::StreamExt;

        if self.stream.is_some() {
            return Err(TransportError::new(
                "transport already has an active request",
            ));
        }
        let response = self
            .client
            .post(&self.generate_url)
            .json(&request)
            .send()
            .await
            .map_err(http_error)?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.map_err(http_error)?;
            return Err(http_response_error(status, &body));
        }
        self.stream = Some(response.bytes_stream().boxed());
        Ok(())
    }

    async fn next_event(&mut self) -> Result<Option<SseEvent>, TransportError> {
        use futures::StreamExt;

        loop {
            if let Some((frame_end, delimiter_len)) = sse_frame_end(&self.buffered) {
                let frame: Vec<u8> = self.buffered.drain(..frame_end).collect();
                self.buffered.drain(..delimiter_len);
                if let Some(event) = parse_sse_frame(&frame)? {
                    return Ok(Some(event));
                }
                continue;
            }

            let chunk = {
                let stream = self
                    .stream
                    .as_mut()
                    .ok_or_else(|| TransportError::new("next_event called before send"))?;
                stream.next().await
            };
            match chunk {
                Some(Ok(chunk)) => self.buffered.extend_from_slice(&chunk),
                Some(Err(error)) => return Err(http_error(error)),
                None => {
                    self.stream = None;
                    if self.buffered.is_empty() {
                        return Ok(None);
                    }
                    return Err(TransportError::new("SSE stream ended in a partial frame"));
                }
            }
        }
    }

    fn cancel(&mut self, _request_id: RequestId) -> Result<(), TransportError> {
        self.stream = None;
        self.buffered.clear();
        Ok(())
    }
}

#[cfg(feature = "http-transport")]
fn http_error(error: reqwest::Error) -> TransportError {
    TransportError::new(format!("HTTP transport error: {error}"))
}

#[cfg(feature = "http-transport")]
fn http_response_error(status: reqwest::StatusCode, body: &str) -> TransportError {
    let detail = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|payload| remote_error_message(&payload));
    match detail {
        Some(detail) => TransportError::new(format!("HTTP {status}: {detail}")),
        None if body.trim().is_empty() => TransportError::new(format!("HTTP {status}")),
        None => TransportError::new(format!("HTTP {status}: {}", body.trim())),
    }
}

fn sse_frame_end(buffer: &[u8]) -> Option<(usize, usize)> {
    for start in 0..buffer.len() {
        let Some(first_len) = line_ending_len(buffer, start) else {
            continue;
        };
        let second_start = start + first_len;
        if let Some(second_len) = line_ending_len(buffer, second_start) {
            return Some((start, first_len + second_len));
        }
    }
    None
}

fn line_ending_len(buffer: &[u8], at: usize) -> Option<usize> {
    match buffer.get(at) {
        Some(b'\r') if buffer.get(at + 1) == Some(&b'\n') => Some(2),
        Some(b'\r') | Some(b'\n') => Some(1),
        _ => None,
    }
}

fn parse_sse_frame(frame: &[u8]) -> Result<Option<SseEvent>, TransportError> {
    let frame = std::str::from_utf8(frame)
        .map_err(|error| TransportError::new(format!("invalid UTF-8 SSE frame: {error}")))?;
    let mut event = None;
    let mut data = Vec::new();
    for line in frame.split(['\r', '\n']) {
        if line.starts_with(':') || line.is_empty() {
            continue;
        }
        let (field, value) = line.split_once(':').unwrap_or((line, ""));
        let value = value.strip_prefix(' ').unwrap_or(value);
        match field {
            "event" => event = Some(value.to_owned()),
            "data" => data.push(value),
            _ => {}
        }
    }
    Ok((!data.is_empty()).then(|| SseEvent {
        event,
        data: data.join("\n"),
    }))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamOutput {
    Token { token: TokenId, text: String },
    Terminal { finish_reason: String },
}

/// Starts request streams without prescribing a network client or clock.
#[derive(Debug)]
pub struct DynamoAdapter<F: TransportFactory> {
    factory: F,
}

impl<F: TransportFactory> DynamoAdapter<F> {
    pub fn new(factory: F) -> Self {
        Self { factory }
    }

    pub async fn start(
        &self,
        request: &Request,
    ) -> Result<RequestStream<F::Transport>, TransportError> {
        let mut transport = self.factory.open()?;
        transport
            .send(TransportRequest::from_request(request))
            .await?;
        Ok(RequestStream {
            transport,
            request_id: request.id,
            telemetry: RequestTelemetry::new(request),
            terminal: false,
            cleaned_up: false,
            outcome: None,
        })
    }
}

/// A request-granularity stream with exactly-once terminal and cleanup state.
pub struct RequestStream<T: Transport> {
    transport: T,
    request_id: RequestId,
    telemetry: RequestTelemetry,
    terminal: bool,
    cleaned_up: bool,
    outcome: Option<trtllm_core::RequestOutcome>,
}

impl<T: Transport> RequestStream<T> {
    pub async fn next_at(&mut self, at_ms: Millis) -> Result<Option<StreamOutput>, TransportError> {
        if self.terminal {
            return Ok(None);
        }

        let event = match self.transport.next_event().await {
            Ok(Some(event)) => event,
            Ok(None) => {
                return self.fail(TransportError::new("stream ended before terminal event"))
            }
            Err(error) => return self.fail(error),
        };

        let decoded = match decode_event(&event) {
            Ok(decoded) => decoded,
            Err(error) => return self.fail(error),
        };
        match decoded {
            DecodedEvent::Token { token, text } => {
                self.telemetry.observe_token(at_ms);
                Ok(Some(StreamOutput::Token { token, text }))
            }
            DecodedEvent::Terminal { finish_reason } => {
                self.terminal = true;
                self.cleaned_up = true;
                self.outcome = Some(self.telemetry.clone().finish(at_ms));
                Ok(Some(StreamOutput::Terminal { finish_reason }))
            }
        }
    }

    pub fn cancel_at(&mut self, at_ms: Millis) -> Result<Option<StreamOutput>, TransportError> {
        if self.terminal {
            return Ok(None);
        }
        let cleanup_result = self.cleanup();
        self.terminal = true;
        self.outcome = Some(self.telemetry.clone().finish(at_ms));
        cleanup_result?;
        Ok(Some(StreamOutput::Terminal {
            finish_reason: "cancelled".into(),
        }))
    }

    pub fn outcome(&self) -> Option<trtllm_core::RequestOutcome> {
        self.outcome
    }

    fn fail<U>(&mut self, error: TransportError) -> Result<U, TransportError> {
        let _ = self.cleanup();
        self.terminal = true;
        Err(error)
    }

    fn cleanup(&mut self) -> Result<(), TransportError> {
        if !self.cleaned_up {
            self.cleaned_up = true;
            return self.transport.cancel(self.request_id);
        }
        Ok(())
    }
}

impl<T: Transport> Drop for RequestStream<T> {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

#[derive(Debug)]
enum DecodedEvent {
    Token { token: TokenId, text: String },
    Terminal { finish_reason: String },
}

fn decode_event(event: &SseEvent) -> Result<DecodedEvent, TransportError> {
    if event.data.trim() == "[DONE]" {
        return Ok(DecodedEvent::Terminal {
            finish_reason: "stop".into(),
        });
    }

    let payload: serde_json::Value = serde_json::from_str(&event.data)
        .map_err(|error| TransportError::new(format!("invalid SSE payload: {error}")))?;
    if payload.get("error").is_some() {
        let message = remote_error_message(&payload)
            .unwrap_or_else(|| "SSE error payload has no message".into());
        return Err(TransportError::new(message));
    }
    if let Some(reason) = payload
        .get("finish_reason")
        .and_then(serde_json::Value::as_str)
    {
        return Ok(DecodedEvent::Terminal {
            finish_reason: reason.into(),
        });
    }
    let token = payload
        .get("token_id")
        .and_then(serde_json::Value::as_u64)
        .and_then(|token| u32::try_from(token).ok())
        .ok_or_else(|| TransportError::new("SSE token payload is missing token_id"))?;
    let text = payload
        .get("text")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| TransportError::new("SSE token payload is missing text"))?
        .to_string();
    Ok(DecodedEvent::Token { token, text })
}

fn remote_error_message(payload: &serde_json::Value) -> Option<String> {
    let error = payload.get("error")?;
    if let Some(message) = error.as_str() {
        return Some(message.into());
    }
    let error = error.as_object()?;
    let message = error.get("message").and_then(serde_json::Value::as_str)?;
    let code = error
        .get("code")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("remote_error");
    Some(format!("{code}: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_multiline_sse_data_and_ignores_metadata() {
        assert_eq!(
            parse_sse_frame(b": keepalive\nevent: token\nid: 7\ndata: first\ndata: second\r\n")
                .expect("frame parses"),
            Some(SseEvent {
                event: Some("token".into()),
                data: "first\nsecond".into(),
            })
        );
    }

    #[test]
    fn recognizes_lf_and_crlf_frame_boundaries() {
        assert_eq!(sse_frame_end(b"data: x\n\nrest"), Some((7, 2)));
        assert_eq!(sse_frame_end(b"data: x\r\n\r\nrest"), Some((7, 4)));
    }

    #[test]
    fn recognizes_cr_only_and_mixed_line_ending_boundaries() {
        assert_eq!(sse_frame_end(b"data: x\r\rrest"), Some((7, 2)));
        assert_eq!(sse_frame_end(b"data: x\r\n\nrest"), Some((7, 3)));
        assert_eq!(sse_frame_end(b"data: x\n\r\nrest"), Some((7, 3)));
    }

    #[test]
    fn parses_cr_only_sse_lines() {
        assert_eq!(
            parse_sse_frame(b"event: token\rdata: first\rdata: second\r").expect("frame parses"),
            Some(SseEvent {
                event: Some("token".into()),
                data: "first\nsecond".into(),
            })
        );
    }

    #[test]
    fn decodes_nested_error_payloads() {
        assert_eq!(
            decode_event(&SseEvent::data(
                r#"{"error":{"code":"invalid_request","message":"bad prompt"}}"#,
            ))
            .expect_err("nested error is terminal")
            .to_string(),
            "transport error: invalid_request: bad prompt"
        );
    }

    #[cfg(feature = "http-transport")]
    #[test]
    fn concrete_http_transport_decodes_chunked_sse_frames() {
        use futures::{executor::block_on, stream, StreamExt};

        let chunks = stream::iter(vec![
            Ok(bytes::Bytes::from_static(
                b"data: {\"token_id\":1,\"text\":\"a\"}\r",
            )),
            Ok(bytes::Bytes::from_static(
                b"\n\r\ndata: {\"finish_reason\":\"eos\"}\n\n",
            )),
        ])
        .boxed();
        let mut transport = HttpTransport {
            client: reqwest::Client::new(),
            generate_url: "http://unused/generate".into(),
            stream: Some(chunks),
            buffered: Vec::new(),
        };

        assert_eq!(
            block_on(transport.next_event()).expect("token frame parses"),
            Some(SseEvent::data(r#"{"token_id":1,"text":"a"}"#)),
        );
        assert_eq!(
            block_on(transport.next_event()).expect("terminal frame parses"),
            Some(SseEvent::data(r#"{"finish_reason":"eos"}"#)),
        );
        transport
            .cancel(RequestId(7))
            .expect("local cancel succeeds");
    }
}
