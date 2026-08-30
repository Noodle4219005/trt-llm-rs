//! Dynamo v1.4.1 LLMEngine adapter over Task 3's owned request stream.
//!
//! Transport, cancellation, terminal-event, and drop lifecycle remain owned by
//! Task 3's RequestStream. This module only converts Dynamo boundary types.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, BackendError, DynamoError, EngineConfig, ErrorType, FinishReason,
    GenerateContext, LLMEngine, LLMEngineOutput, LlmRegistration, PreprocessedRequest, Worker,
};
use futures::stream::BoxStream;
use futures::StreamExt;
use trtllm_core::{Request, RequestId, SamplingParams};

use crate::config::DynamoEngineConfig;
use crate::{DynamoAdapter, StreamOutput, TransportError, TransportFactory};

pub struct DynamoLlmEngine<F: TransportFactory> {
    adapter: DynamoAdapter<F>,
    config: DynamoEngineConfig,
    next_request_id: AtomicU64,
    started: AtomicBool,
    active: Arc<AtomicUsize>,
}

impl<F: TransportFactory> DynamoLlmEngine<F> {
    pub fn new(factory: F, config: DynamoEngineConfig) -> Self {
        Self {
            adapter: DynamoAdapter::new(factory),
            config,
            next_request_id: AtomicU64::new(0),
            started: AtomicBool::new(false),
            active: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn convert_request(&self, request: &PreprocessedRequest) -> Result<Request, DynamoError> {
        reject_unsupported_request(request)?;

        let mut extra = serde_json::Map::new();
        if let Some(stop) = &request.stop_conditions.stop {
            extra.insert("stop".into(), serde_json::json!(stop));
        }
        let mut stop_token_ids = request
            .stop_conditions
            .stop_token_ids
            .clone()
            .unwrap_or_default();
        if let Some(hidden) = &request.stop_conditions.stop_token_ids_hidden {
            stop_token_ids.extend(hidden.iter().copied());
        }
        if !stop_token_ids.is_empty() {
            extra.insert("stop_token_ids".into(), serde_json::json!(stop_token_ids));
        }
        if let Some(min_tokens) = request.stop_conditions.min_tokens {
            extra.insert("min_tokens".into(), serde_json::json!(min_tokens));
        }
        if let Some(max_thinking_tokens) = request.stop_conditions.max_thinking_tokens {
            extra.insert(
                "thinking_token_budget".into(),
                serde_json::json!(max_thinking_tokens),
            );
        }
        if let Some(value) = request.sampling_options.repetition_penalty {
            extra.insert("repetition_penalty".into(), serde_json::json!(value));
        }
        if let Some(value) = request.sampling_options.presence_penalty {
            extra.insert("presence_penalty".into(), serde_json::json!(value));
        }
        if let Some(value) = request.sampling_options.frequency_penalty {
            extra.insert("frequency_penalty".into(), serde_json::json!(value));
        }
        if let Some(value) = request.sampling_options.min_p {
            extra.insert("min_p".into(), serde_json::json!(value));
        }
        if let Some(value) = request.sampling_options.include_stop_str_in_output {
            extra.insert(
                "include_stop_str_in_output".into(),
                serde_json::json!(value),
            );
        }
        if let Some(value) = request.output_options.skip_special_tokens {
            extra.insert("skip_special_tokens".into(), serde_json::json!(value));
        }
        // No `spaces_between_special_tokens` mapping: Dynamo v1.4.1's
        // OutputOptions does not have that field
        // (third_party/dynamo/lib/llm/src/protocols/common.rs:612-635 carries
        // only logprobs, prompt_logprobs, skip_special_tokens, formatted_prompt
        // and return_tokens_as_token_ids). The Python worker still accepts it
        // on the wire for direct clients; it is simply unreachable from a
        // Dynamo request, so there is no control being silently dropped here.

        let seed = match request.sampling_options.seed {
            Some(seed) if seed < 0 => {
                return Err(invalid_request(
                    "sampling_options.seed must be non-negative",
                ))
            }
            Some(seed) => Some(u64::try_from(seed).expect("non-negative i64 fits u64")),
            None => None,
        };

        let arrival_ms = request.request_timestamp_ms.unwrap_or_else(now_ms);
        Ok(Request {
            id: RequestId(self.next_request_id.fetch_add(1, Ordering::Relaxed)),
            prompt: request.token_ids.clone(),
            params: SamplingParams {
                max_tokens: request
                    .stop_conditions
                    .max_tokens
                    .unwrap_or(self.config.default_max_tokens),
                temperature: request.sampling_options.temperature.unwrap_or(0.0),
                top_p: request.sampling_options.top_p.unwrap_or(1.0),
                top_k: request.sampling_options.top_k.unwrap_or(-1),
                ignore_eos: request.stop_conditions.ignore_eos.unwrap_or(false),
                seed,
                extra,
            },
            arrival_ms,
            ttft_deadline_ms: f64::INFINITY,
            prefill_worker: None,
            decode_worker: None,
        })
    }
}

/// Construct a Worker instead of reproducing Dynamo's runtime, discovery,
/// endpoint registration, metrics, or shutdown lifecycle.
pub fn worker_with_factory<F: TransportFactory>(factory: F, config: DynamoEngineConfig) -> Worker {
    Worker::new(
        Arc::new(DynamoLlmEngine::new(factory, config.clone())),
        config.worker_config(),
    )
}

pub fn run_with_factory<F: TransportFactory>(
    factory: F,
    config: DynamoEngineConfig,
) -> anyhow::Result<()> {
    dynamo_backend_common::run(
        Arc::new(DynamoLlmEngine::new(factory, config.clone())),
        config.worker_config(),
    )
}

#[async_trait]
impl<F: TransportFactory> LLMEngine for DynamoLlmEngine<F> {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        if self.started.swap(true, Ordering::AcqRel) {
            return Err(engine_error("engine already started"));
        }
        Ok(EngineConfig {
            model: self.config.model.clone(),
            served_model_name: self.config.served_model_name.clone(),
            llm: Some(LlmRegistration::default()),
            ..Default::default()
        })
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        if !self.started.load(Ordering::Acquire) {
            return Err(engine_error("generate called before start"));
        }
        let stream = self
            .adapter
            .start(&self.convert_request(&request)?)
            .await
            .map_err(transport_error)?;
        self.active.fetch_add(1, Ordering::AcqRel);
        let guard = ActiveGuard(self.active.clone());
        Ok(futures::stream::unfold(
            (stream, ctx, guard),
            |(mut stream, ctx, guard)| async move {
                let next = if ctx.is_stopped() {
                    stream.cancel_at(now_ms())
                } else {
                    stream.next_at(now_ms()).await
                };
                match next {
                    Ok(Some(output)) => Some((Ok(to_dynamo_output(output)), (stream, ctx, guard))),
                    Ok(None) => None,
                    Err(error) => Some((Err(transport_error(error)), (stream, ctx, guard))),
                }
            },
        )
        .boxed())
    }

    async fn abort(&self, _ctx: Arc<dyn AsyncEngineContext>) {
        // The returned stream observes ctx.is_stopped() and hands cancellation
        // to Task 3's exactly-once RequestStream cleanup.
    }

    async fn is_quiescent(&self) -> Result<Option<bool>, DynamoError> {
        Ok(Some(self.active.load(Ordering::Acquire) == 0))
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        self.started.store(false, Ordering::Release);
        Ok(())
    }

    async fn health_check_payload(&self) -> Result<Option<serde_json::Value>, DynamoError> {
        Ok(Some(
            serde_json::json!({"model": self.config.model, "token_ids": []}),
        ))
    }

    async fn supported_controls(&self) -> Result<Vec<String>, DynamoError> {
        Ok(vec!["health".into()])
    }

    async fn engine_control(
        &self,
        control: String,
        _body: serde_json::Value,
    ) -> Result<serde_json::Value, DynamoError> {
        if control == "health" {
            Ok(serde_json::json!({
                "status": "ok",
                "active_requests": self.active.load(Ordering::Acquire),
            }))
        } else {
            Ok(serde_json::json!({
                "status": "error",
                "message": format!("unsupported engine control: {control}"),
            }))
        }
    }

    async fn supported_updates(&self) -> Result<Vec<String>, DynamoError> {
        Ok(Vec::new())
    }
}

struct ActiveGuard(Arc<AtomicUsize>);

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

fn to_dynamo_output(output: StreamOutput) -> LLMEngineOutput {
    match output {
        StreamOutput::Token { token, text } => LLMEngineOutput {
            token_ids: vec![token],
            tokens: Some(vec![Some(text.clone())]),
            text: Some(text),
            ..Default::default()
        },
        StreamOutput::Terminal { finish_reason } if finish_reason == "cancelled" => {
            LLMEngineOutput::cancelled()
        }
        StreamOutput::Terminal { finish_reason } if finish_reason == "length" => {
            LLMEngineOutput::length()
        }
        StreamOutput::Terminal { finish_reason } => match finish_reason.as_str() {
            "eos" => LLMEngineOutput {
                finish_reason: Some(FinishReason::EoS),
                ..Default::default()
            },
            "stop" => LLMEngineOutput::stop(),
            "content_filter" => LLMEngineOutput {
                finish_reason: Some(FinishReason::ContentFilter),
                ..Default::default()
            },
            "error" => LLMEngineOutput::error("worker reported an error".into()),
            "timeout" => LLMEngineOutput::error("worker timed out".into()),
            other => LLMEngineOutput::error(format!("unsupported finish reason: {other}")),
        },
    }
}

fn reject_unsupported_request(request: &PreprocessedRequest) -> Result<(), DynamoError> {
    if request.prompt_embeds.is_some()
        || request.multi_modal_data.is_some()
        || request.multi_modal_uuids.is_some()
        || request.mm_routing_info.is_some()
        || request.prefill_result.is_some()
        || request.encoder_result.is_some()
        || request.bootstrap_info.is_some()
        || request.extra_args.is_some()
        || request.mm_processor_kwargs.is_some()
    {
        return Err(invalid_request(
            "multimodal, disaggregated, or backend extra request fields are not supported by the HTTP worker",
        ));
    }
    if request
        .stop_conditions
        .stop_token_ids_visible
        .as_ref()
        .is_some_and(|ids| !ids.is_empty())
    {
        return Err(invalid_request(
            "stop_token_ids_visible is not supported by TensorRT-LLM worker",
        ));
    }
    if request.sampling_options.n.is_some_and(|n| n != 1)
        || request
            .sampling_options
            .best_of
            .is_some_and(|best_of| best_of != 1)
    {
        return Err(invalid_request(
            "only one output sequence is supported by the streaming worker",
        ));
    }
    if request.sampling_options.use_beam_search == Some(true) {
        return Err(invalid_request(
            "beam search is not supported by the streaming worker",
        ));
    }
    if request.sampling_options.length_penalty.is_some()
        || request.sampling_options.guided_decoding.is_some()
    {
        return Err(invalid_request(
            "length penalty and guided decoding are not supported by the streaming worker",
        ));
    }
    if request.output_options.logprobs.is_some()
        || request.output_options.prompt_logprobs.is_some()
        || request.output_options.formatted_prompt.is_some()
        || request.output_options.return_tokens_as_token_ids.is_some()
    {
        return Err(invalid_request(
            "requested logprob or formatted-output controls are not supported by the streaming worker",
        ));
    }
    Ok(())
}

fn now_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
        * 1000.0
}

fn engine_error(message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::EngineShutdown))
        .message(message)
        .build()
}

fn invalid_request(message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::InvalidArgument))
        .message(message)
        .build()
}

fn transport_error(error: TransportError) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::Unknown))
        .message(error.to_string())
        .build()
}

#[cfg(test)]
mod type_tests {
    use super::*;

    // Break caught: losing the v1.4.1 trait implementation or Task 3's
    // Send+Sync factory guarantee must fail this feature type contract.
    fn assert_llm_engine<T: LLMEngine>() {}

    #[test]
    fn task3_factory_backed_adapter_implements_dynamo_llm_engine() {
        assert_llm_engine::<DynamoLlmEngine<TestFactory>>();
    }

    struct TestFactory;
    struct TestTransport;

    impl crate::TransportFactory for TestFactory {
        type Transport = TestTransport;
        fn open(&self) -> Result<Self::Transport, TransportError> {
            Ok(TestTransport)
        }
    }

    #[async_trait]
    impl crate::Transport for TestTransport {
        async fn send(&mut self, _request: crate::TransportRequest) -> Result<(), TransportError> {
            Ok(())
        }
        async fn next_event(&mut self) -> Result<Option<crate::SseEvent>, TransportError> {
            Ok(None)
        }
        fn cancel(&mut self, _request_id: RequestId) -> Result<(), TransportError> {
            Ok(())
        }
    }

    fn request_from_json(value: serde_json::Value) -> PreprocessedRequest {
        serde_json::from_value(value).expect("valid preprocessed request")
    }

    fn test_engine() -> DynamoLlmEngine<TestFactory> {
        DynamoLlmEngine::new(TestFactory, DynamoEngineConfig::default())
    }

    #[test]
    fn converts_supported_stop_and_output_controls_to_wire_extras() {
        let request = request_from_json(serde_json::json!({
            "token_ids": [1, 2, 3],
            "stop_conditions": {
                "max_tokens": 9,
                "stop": ["<END>"],
                "stop_token_ids": [9],
                "stop_token_ids_hidden": [10],
                "min_tokens": 2,
                "ignore_eos": false
            },
            "sampling_options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 8,
                "seed": 4,
                "repetition_penalty": 1.1,
                "presence_penalty": 0.2,
                "frequency_penalty": 0.3,
                "include_stop_str_in_output": true
            },
            "output_options": {
                "skip_special_tokens": false
            }
        }));

        let converted = test_engine()
            .convert_request(&request)
            .expect("supported controls convert");

        assert_eq!(converted.params.max_tokens, 9);
        assert_eq!(converted.params.temperature, 0.7);
        assert_eq!(converted.params.top_p, 0.9);
        assert_eq!(converted.params.top_k, 8);
        assert_eq!(converted.params.seed, Some(4));
        assert_eq!(converted.params.ignore_eos, false);
        assert_eq!(
            converted.params.extra,
            serde_json::json!({
                "stop": ["<END>"],
                "stop_token_ids": [9, 10],
                "min_tokens": 2,
                // f32 in Dynamo's SamplingOptions, so the wire carries the
                // widened f32, not the f64 these literals would otherwise be.
                "repetition_penalty": 1.1f32,
                "presence_penalty": 0.2f32,
                "frequency_penalty": 0.3f32,
                "include_stop_str_in_output": true,
                "skip_special_tokens": false
            })
            .as_object()
            .expect("object extras")
            .clone()
        );
    }

    #[test]
    fn rejects_multiple_outputs_before_starting_transport() {
        let request = request_from_json(serde_json::json!({
            "token_ids": [1],
            "sampling_options": {"n": 2}
        }));

        let error = test_engine()
            .convert_request(&request)
            .expect_err("multi-output request must fail closed");

        assert!(format!("{error}").contains("n"));
    }

    #[test]
    fn preserves_all_supported_terminal_reasons() {
        use dynamo_backend_common::FinishReason;

        assert_eq!(
            to_dynamo_output(StreamOutput::Terminal {
                finish_reason: "eos".into(),
            })
            .finish_reason,
            Some(FinishReason::EoS)
        );
        assert_eq!(
            to_dynamo_output(StreamOutput::Terminal {
                finish_reason: "content_filter".into(),
            })
            .finish_reason,
            Some(FinishReason::ContentFilter)
        );
        assert!(matches!(
            to_dynamo_output(StreamOutput::Terminal {
                finish_reason: "unknown".into(),
            })
            .finish_reason,
            Some(FinishReason::Error(_))
        ));
    }
}
