use trtllm_core::{Millis, Request, RequestOutcome};

/// Request-scoped observations collected by the serving adapter.
///
/// Time is passed in by the caller: the adapter can use a production monotonic
/// clock while deterministic tests provide fixed timestamps.
#[derive(Clone, Debug)]
pub struct RequestTelemetry {
    request: Request,
    first_token_ms: Option<Millis>,
    last_token_ms: Option<Millis>,
    output_tokens: u32,
}

impl RequestTelemetry {
    pub fn new(request: &Request) -> Self {
        Self {
            request: request.clone(),
            first_token_ms: None,
            last_token_ms: None,
            output_tokens: 0,
        }
    }

    pub fn observe_token(&mut self, at_ms: Millis) {
        self.first_token_ms.get_or_insert(at_ms);
        self.last_token_ms = Some(at_ms);
        self.output_tokens += 1;
    }

    /// Build the shared scoring input without duplicating goodput logic.
    pub fn finish(self, at_ms: Millis) -> RequestOutcome {
        let first_token_ms = self.first_token_ms.unwrap_or(at_ms);
        RequestOutcome {
            id: self.request.id,
            arrival_ms: self.request.arrival_ms,
            first_token_ms,
            last_token_ms: self.last_token_ms.unwrap_or(at_ms),
            prompt_tokens: self.request.prompt_len() as u32,
            output_tokens: self.output_tokens,
            requested_tokens: self.request.params.max_tokens,
        }
    }
}
