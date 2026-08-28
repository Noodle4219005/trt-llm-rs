use serde::{Deserialize, Serialize};

use crate::ids::{RequestId, WorkerId};
use crate::slo::{Slo, Verdict};
use crate::{Millis, TokenId};

/// Sampling parameters carried end to end. Only the fields the scheduler needs
/// to reason about capacity live here; anything backend specific travels in
/// `extra` so a new engine can be added without touching the control plane.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamplingParams {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub ignore_eos: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_tokens: 200,
            temperature: 0.0,
            top_p: 1.0,
            top_k: -1,
            ignore_eos: true,
            seed: None,
            extra: serde_json::Map::new(),
        }
    }
}

/// Which half of a disaggregated deployment a unit of work belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Phase {
    Prefill,
    Decode,
}

impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Prefill => "prefill",
            Phase::Decode => "decode",
        }
    }
}

/// One inference request as the control plane sees it.
#[derive(Clone, Debug)]
pub struct Request {
    pub id: RequestId,
    pub prompt: Vec<TokenId>,
    pub params: SamplingParams,
    /// When the frontend accepted the request.
    pub arrival_ms: Millis,
    /// Absolute deadline for the first token, derived from the SLO at arrival.
    pub ttft_deadline_ms: Millis,
    /// Set once the router has bound the request to a prefill worker.
    pub prefill_worker: Option<WorkerId>,
    /// Set once the router has bound the request to a decode worker.
    pub decode_worker: Option<WorkerId>,
}

impl Request {
    pub fn new(
        id: RequestId,
        prompt: Vec<TokenId>,
        params: SamplingParams,
        arrival_ms: Millis,
        slo: &Slo,
    ) -> Self {
        Self {
            id,
            ttft_deadline_ms: arrival_ms + slo.ttft_ms,
            prompt,
            params,
            arrival_ms,
            prefill_worker: None,
            decode_worker: None,
        }
    }

    pub fn prompt_len(&self) -> usize {
        self.prompt.len()
    }

    /// Slack against the first-token deadline at `now`. Negative means the
    /// request is already doomed to be late; the scheduler uses that to stop
    /// spending head-of-line capacity on it.
    pub fn slack_ms(&self, now: Millis) -> Millis {
        self.ttft_deadline_ms - now
    }
}

/// Everything needed to score one finished request.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct RequestOutcome {
    pub id: RequestId,
    pub arrival_ms: Millis,
    pub first_token_ms: Millis,
    pub last_token_ms: Millis,
    pub prompt_tokens: u32,
    pub output_tokens: u32,
    pub requested_tokens: u32,
}

impl RequestOutcome {
    pub fn ttft_ms(&self) -> f64 {
        self.first_token_ms - self.arrival_ms
    }

    /// Mean inter-token latency: the gap between the first and the last token
    /// divided by the number of *gaps*, which is one fewer than the number of
    /// tokens. Dividing by `output_tokens` instead is the classic off-by-one
    /// that flatters ITL by ~0.5 % at OSL 200 and by 10 % at OSL 10.
    pub fn mean_itl_ms(&self) -> f64 {
        if self.output_tokens <= 1 {
            return 0.0;
        }
        (self.last_token_ms - self.first_token_ms) / f64::from(self.output_tokens - 1)
    }

    pub fn verdict(&self, slo: &Slo) -> Verdict {
        if self.output_tokens < self.requested_tokens {
            return Verdict::Incomplete;
        }
        let late = self.ttft_ms() > slo.ttft_ms;
        let slow = self.mean_itl_ms() > slo.itl_ms;
        match (late, slow) {
            (false, false) => Verdict::Good,
            (true, false) => Verdict::LateFirstToken,
            (false, true) => Verdict::SlowTokens,
            (true, true) => Verdict::LateAndSlow,
        }
    }
}
