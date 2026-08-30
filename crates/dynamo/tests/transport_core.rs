use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use trtllm_core::{GoodputReport, Request, RequestId, SamplingParams, Slo};
use trtllm_dynamo::{
    DynamoAdapter, SseEvent, StreamOutput, Transport, TransportError, TransportFactory,
    TransportRequest,
};

#[derive(Clone, Default)]
struct MockFactory {
    events: VecDeque<Result<Option<SseEvent>, TransportError>>,
    cancelled: Arc<Mutex<Vec<RequestId>>>,
    cancel_error: Option<TransportError>,
}

struct MockTransport {
    events: VecDeque<Result<Option<SseEvent>, TransportError>>,
    cancelled: Arc<Mutex<Vec<RequestId>>>,
    cancel_error: Option<TransportError>,
}

impl MockFactory {
    fn with_events(events: Vec<Result<Option<SseEvent>, TransportError>>) -> Self {
        Self {
            events: events.into_iter().collect(),
            cancelled: Arc::new(Mutex::new(Vec::new())),
            cancel_error: None,
        }
    }

    fn with_cancel_error(
        events: Vec<Result<Option<SseEvent>, TransportError>>,
        cancel_error: TransportError,
    ) -> Self {
        Self {
            events: events.into_iter().collect(),
            cancelled: Arc::new(Mutex::new(Vec::new())),
            cancel_error: Some(cancel_error),
        }
    }

    fn cancelled(&self) -> Vec<RequestId> {
        self.cancelled.lock().expect("mock state lock").clone()
    }
}

impl TransportFactory for MockFactory {
    type Transport = MockTransport;

    fn open(&self) -> Result<Self::Transport, TransportError> {
        Ok(MockTransport {
            events: self.events.clone(),
            cancelled: Arc::clone(&self.cancelled),
            cancel_error: self.cancel_error.clone(),
        })
    }
}

#[async_trait]
impl Transport for MockTransport {
    async fn send(&mut self, _request: TransportRequest) -> Result<(), TransportError> {
        Ok(())
    }

    async fn next_event(&mut self) -> Result<Option<SseEvent>, TransportError> {
        self.events.pop_front().unwrap_or(Ok(None))
    }

    fn cancel(&mut self, request_id: RequestId) -> Result<(), TransportError> {
        self.cancelled
            .lock()
            .expect("mock state lock")
            .push(request_id);
        if let Some(error) = &self.cancel_error {
            return Err(error.clone());
        }
        Ok(())
    }
}

fn request() -> Request {
    Request::new(
        RequestId(7),
        vec![11, 12],
        SamplingParams {
            max_tokens: 2,
            temperature: 0.7,
            ..Default::default()
        },
        1_000.0,
        &Slo::default(),
    )
}

#[test]
fn serializes_the_core_request_without_reimplementing_sampling() {
    let request = request();
    let wire = TransportRequest::from_request(&request);

    assert_eq!(
        serde_json::to_value(wire).expect("wire request serializes"),
        serde_json::json!({
            "request_id": "req-7",
            "prompt_token_ids": [11, 12],
            "sampling": {
                "max_tokens": 2,
                // f32 on the wire: 0.7f32 widens to 0.699999988079071, and the
                // expectation has to say so rather than hide it behind an f64
                // literal that never appears in a real payload.
                "temperature": 0.7f32,
                "top_p": 1.0f32,
                "top_k": -1,
                "ignore_eos": true
                // no "extra": SamplingParams skips it when empty, and the
                // worker treats absent as {}.
            }
        })
    );
}

#[test]
fn converts_sse_token_payloads_and_done_markers_to_stream_outputs() {
    let factory = MockFactory::with_events(vec![
        Ok(Some(SseEvent::data(r#"{"token_id":42,"text":"hi"}"#))),
        Ok(Some(SseEvent::data("[DONE]"))),
    ]);
    let adapter = DynamoAdapter::new(factory);
    let mut stream =
        futures::executor::block_on(adapter.start(&request())).expect("request starts");

    assert_eq!(
        futures::executor::block_on(stream.next_at(1_010.0)).expect("token event"),
        Some(StreamOutput::Token {
            token: 42,
            text: "hi".into(),
        })
    );
    assert_eq!(
        futures::executor::block_on(stream.next_at(1_020.0)).expect("done event"),
        Some(StreamOutput::Terminal {
            finish_reason: "stop".into(),
        })
    );
}

#[test]
fn emits_only_one_terminal_event_and_makes_a_scoreable_outcome() {
    let factory = MockFactory::with_events(vec![
        Ok(Some(SseEvent::data(r#"{"token_id":1,"text":"a"}"#))),
        Ok(Some(SseEvent::data("[DONE]"))),
        Ok(Some(SseEvent::data("[DONE]"))),
    ]);
    let adapter = DynamoAdapter::new(factory);
    let mut stream =
        futures::executor::block_on(adapter.start(&request())).expect("request starts");

    assert!(matches!(
        futures::executor::block_on(stream.next_at(1_010.0)),
        Ok(Some(StreamOutput::Token { .. }))
    ));
    assert!(matches!(
        futures::executor::block_on(stream.next_at(1_020.0)),
        Ok(Some(StreamOutput::Terminal { .. }))
    ));
    assert_eq!(
        futures::executor::block_on(stream.next_at(1_030.0)).expect("terminal is final"),
        None
    );

    let outcome = stream.outcome().expect("terminal yields telemetry");
    assert_eq!(outcome.id, RequestId(7));
    assert_eq!(outcome.prompt_tokens, 2);
    assert_eq!(outcome.output_tokens, 1);
    let report = GoodputReport::from_outcomes(&[outcome], 1.0, &Slo::default());
    assert_eq!(report.total_requests, 1);
    assert_eq!(report.incomplete, 1);
}

#[test]
fn cancelling_emits_one_terminal_event_and_cleans_up_the_request() {
    let factory = MockFactory::default();
    let adapter = DynamoAdapter::new(factory.clone());
    let mut stream =
        futures::executor::block_on(adapter.start(&request())).expect("request starts");

    assert_eq!(
        stream.cancel_at(1_010.0).expect("cancellation succeeds"),
        Some(StreamOutput::Terminal {
            finish_reason: "cancelled".into(),
        })
    );
    assert_eq!(
        futures::executor::block_on(stream.next_at(1_020.0)).expect("terminal is final"),
        None
    );
    drop(stream);

    assert_eq!(factory.cancelled(), vec![RequestId(7)]);
}

#[test]
fn cloned_factory_shares_cancellation_observation() {
    let factory = MockFactory::default();
    let adapter = DynamoAdapter::new(factory.clone());
    let stream = futures::executor::block_on(adapter.start(&request())).expect("request starts");

    drop(stream);

    assert_eq!(factory.cancelled(), vec![RequestId(7)]);
}

#[test]
fn failed_stream_does_not_retry_a_side_effecting_cancel_error_on_drop() {
    let factory = MockFactory::with_cancel_error(
        vec![Err(TransportError::new("worker disconnected"))],
        TransportError::new("cancel failed"),
    );
    let adapter = DynamoAdapter::new(factory.clone());
    let mut stream =
        futures::executor::block_on(adapter.start(&request())).expect("request starts");

    assert_eq!(
        futures::executor::block_on(stream.next_at(1_010.0)).expect_err("worker error propagates"),
        TransportError::new("worker disconnected")
    );
    drop(stream);

    assert_eq!(factory.cancelled(), vec![RequestId(7)]);
}

#[test]
fn cancelling_does_not_retry_a_side_effecting_cancel_error_on_drop() {
    let factory = MockFactory::with_cancel_error(vec![], TransportError::new("cancel failed"));
    let adapter = DynamoAdapter::new(factory.clone());
    let mut stream =
        futures::executor::block_on(adapter.start(&request())).expect("request starts");

    assert_eq!(
        stream
            .cancel_at(1_010.0)
            .expect_err("cancel error propagates"),
        TransportError::new("cancel failed")
    );
    assert_eq!(
        futures::executor::block_on(stream.next_at(1_020.0))
            .expect("cancel failure leaves the stream terminal"),
        None
    );
    drop(stream);

    assert_eq!(factory.cancelled(), vec![RequestId(7)]);
}

#[test]
fn dropping_an_unfinished_stream_cancels_once() {
    let factory = MockFactory::default();
    let adapter = DynamoAdapter::new(factory.clone());
    let stream = futures::executor::block_on(adapter.start(&request())).expect("request starts");

    drop(stream);

    assert_eq!(factory.cancelled(), vec![RequestId(7)]);
}

#[test]
fn propagates_transport_errors_and_releases_the_request() {
    let factory = MockFactory::with_events(vec![Err(TransportError::new("worker disconnected"))]);
    let adapter = DynamoAdapter::new(factory.clone());
    let mut stream =
        futures::executor::block_on(adapter.start(&request())).expect("request starts");

    assert_eq!(
        futures::executor::block_on(stream.next_at(1_010.0)).expect_err("worker error propagates"),
        TransportError::new("worker disconnected")
    );
    drop(stream);

    assert_eq!(factory.cancelled(), vec![RequestId(7)]);
}

#[test]
fn cancelling_completed_stream_emits_no_second_terminal_event() {
    let factory = MockFactory::with_events(vec![Ok(Some(SseEvent::data("[DONE]")))]);
    let adapter = DynamoAdapter::new(factory.clone());
    let mut stream =
        futures::executor::block_on(adapter.start(&request())).expect("request starts");

    assert_eq!(
        futures::executor::block_on(stream.next_at(1_010.0)).expect("done event"),
        Some(StreamOutput::Terminal {
            finish_reason: "stop".into(),
        })
    );
    assert_eq!(
        stream.cancel_at(1_020.0).expect("cancel is idempotent"),
        None
    );
    assert_eq!(
        futures::executor::block_on(stream.next_at(1_030.0)).expect("terminal is final"),
        None
    );
    drop(stream);

    assert!(factory.cancelled().is_empty());
}

#[test]
fn decoder_errors_enter_terminal_cleanup_state() {
    let factory = MockFactory::with_events(vec![Ok(Some(SseEvent::data("{not-json")))]);
    let adapter = DynamoAdapter::new(factory.clone());
    let mut stream =
        futures::executor::block_on(adapter.start(&request())).expect("request starts");

    assert!(futures::executor::block_on(stream.next_at(1_010.0)).is_err());
    assert_eq!(
        futures::executor::block_on(stream.next_at(1_020.0)).expect("decoder error is terminal"),
        None
    );
    drop(stream);

    assert_eq!(factory.cancelled(), vec![RequestId(7)]);
}

#[test]
fn application_errors_enter_terminal_cleanup_state() {
    let factory = MockFactory::with_events(vec![Ok(Some(SseEvent::data(
        r#"{"error":"worker rejected request"}"#,
    )))]);
    let adapter = DynamoAdapter::new(factory.clone());
    let mut stream =
        futures::executor::block_on(adapter.start(&request())).expect("request starts");

    assert!(futures::executor::block_on(stream.next_at(1_010.0)).is_err());
    assert_eq!(
        futures::executor::block_on(stream.next_at(1_020.0))
            .expect("application error is terminal"),
        None
    );
    drop(stream);

    assert_eq!(factory.cancelled(), vec![RequestId(7)]);
}

#[test]
fn two_streams_can_be_started_without_borrowing_adapter() {
    // Take a reference: the bounds still apply to T, but asserting twice about
    // the same stream must not consume it.
    fn assert_static<T: 'static>(_: &T) {}
    fn assert_send<T: Send>(_: &T) {}

    let adapter = DynamoAdapter::new(MockFactory::default());
    let first =
        futures::executor::block_on(adapter.start(&request())).expect("first stream starts");
    let second =
        futures::executor::block_on(adapter.start(&request())).expect("second stream starts");

    assert_static(&first);
    assert_static(&second);
    assert_send(&first);
    assert_send(&second);
}
