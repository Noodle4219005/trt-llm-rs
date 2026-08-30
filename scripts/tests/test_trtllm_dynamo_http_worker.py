#!/usr/bin/env python3
"""Pure-Python protocol tests for trtllm_dynamo_http_worker.py.

The production worker loads FastAPI and TensorRT-LLM at runtime.  This test
installs only the tiny FastAPI surface the worker uses and passes a fake LLM,
so it never imports TensorRT-LLM or initializes CUDA.
"""

import asyncio
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path


WORKER = Path(__file__).parents[1] / "trtllm_dynamo_http_worker.py"


class FakeFastAPI:
    # This double is deliberately more permissive than real FastAPI, so it
    # cannot catch an invalid response annotation on its own (job 314676 did).
    # It does record the decorator kwargs, which lets a test assert the one
    # constraint the real dependency imposes.
    def __init__(self, **_kwargs):
        self.routes = {}
        self.route_kwargs = {}

    def get(self, path, **kwargs):
        return self._route("GET", path, kwargs)

    def post(self, path, **kwargs):
        return self._route("POST", path, kwargs)

    def _route(self, method, path, kwargs):
        def register(handler):
            self.routes[(method, path)] = handler
            self.route_kwargs[(method, path)] = kwargs
            return handler
        return register


class FakeStreamingResponse:
    def __init__(self, body_iterator, **_kwargs):
        self.body_iterator = body_iterator


class FakeJSONResponse:
    def __init__(self, content, status_code=200, **_kwargs):
        self.content = content
        self.status_code = status_code


class FakeRequest:
    def __init__(self, payload, disconnected=False):
        self.payload = payload
        self.disconnected = disconnected

    async def json(self):
        return self.payload

    async def is_disconnected(self):
        return self.disconnected


class FakePreprocessedInputs:
    def __init__(self, prompt_token_ids):
        self.prompt_token_ids = prompt_token_ids


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeCompletion:
    def __init__(self, token_ids, text, finish_reason=None):
        self.token_ids_diff = token_ids
        self.text_diff = text
        self.finish_reason = finish_reason


class FakeOutput:
    def __init__(self, completion, error=None):
        self.outputs = [completion]
        self.error = error


class FakeRequestOutput:
    def __init__(self, outputs):
        self.outputs = outputs
        self.finished = False
        self.aborted = False

    def abort(self):
        self.aborted = True
        self.finished = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.aborted or not self.outputs:
            self.finished = True
            raise StopAsyncIteration
        output = self.outputs.pop(0)
        if output.outputs[0].finish_reason:
            self.finished = True
        return output


class FakeLLM:
    def __init__(self, request_output=None, error=None):
        self.request_output = request_output
        self.error = error
        self.calls = []

    def generate_async(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.request_output


def load_worker():
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = FakeFastAPI
    fastapi.Request = object
    responses = types.ModuleType("fastapi.responses")
    responses.JSONResponse = FakeJSONResponse
    responses.StreamingResponse = FakeStreamingResponse
    previous = {name: sys.modules.get(name) for name in ("fastapi", "fastapi.responses")}
    sys.modules["fastapi"] = fastapi
    sys.modules["fastapi.responses"] = responses
    try:
        spec = importlib.util.spec_from_file_location("tested_worker", WORKER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, prior in previous.items():
            if prior is None:
                del sys.modules[name]
            else:
                sys.modules[name] = prior


async def collect_sse(response):
    return [json.loads(frame.removeprefix("data: ").strip())
            async for frame in response.body_iterator]


class WorkerProtocolTests(unittest.TestCase):
    def setUp(self):
        self.worker = load_worker()

    def test_generate_maps_supported_request_and_emits_tokens_then_finish(self):
        request_output = FakeRequestOutput([
            FakeOutput(FakeCompletion([17, 18], "hi", "length")),
        ])
        llm = FakeLLM(request_output)
        app = self.worker.create_app(llm, FakePreprocessedInputs, FakeSamplingParams)
        response = asyncio.run(app.routes[("POST", "/generate")](FakeRequest({
            "request_id": "request-7",
            "prompt_token_ids": [1, 2, 3],
            "sampling": {
                "max_tokens": 4,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": -1,
                "ignore_eos": False,
                "seed": 9,
                "extra": {"min_tokens": 2, "stop_token_ids": [99]},
            },
        })))

        frames = asyncio.run(collect_sse(response))

        self.assertEqual(frames, [
            {"token_id": 17, "text": ""},
            {"token_id": 18, "text": "hi"},
            {"finish_reason": "length"},
        ])
        self.assertEqual(llm.calls[0]["inputs"].prompt_token_ids, [1, 2, 3])
        self.assertTrue(llm.calls[0]["streaming"])
        self.assertEqual(llm.calls[0]["sampling_params"].kwargs, {
            "max_tokens": 4, "temperature": 0.7, "top_p": 0.9,
            "top_k": None, "ignore_eos": False, "seed": 9,
            "min_tokens": 2, "stop_token_ids": [99],
        })

    def test_generate_rejects_unknown_extra_without_starting_llm(self):
        llm = FakeLLM()
        app = self.worker.create_app(llm, FakePreprocessedInputs, FakeSamplingParams)
        response = asyncio.run(app.routes[("POST", "/generate")](FakeRequest({
            "request_id": "request-8", "prompt_token_ids": [1],
            "sampling": {"max_tokens": 1, "temperature": 0.0, "top_p": 1.0,
                         "top_k": -1, "ignore_eos": False,
                         "extra": {"unsupported_control": True}},
        })))

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.content["error"]["code"], "unsupported_sampling_extra")
        self.assertEqual(llm.calls, [])

    def test_generate_maps_stop_and_output_controls(self):
        request_output = FakeRequestOutput([
            FakeOutput(FakeCompletion([21], "done", "stop")),
        ])
        llm = FakeLLM(request_output)
        app = self.worker.create_app(llm, FakePreprocessedInputs, FakeSamplingParams)
        response = asyncio.run(app.routes[("POST", "/generate")](FakeRequest({
            "request_id": "request-10",
            "prompt_token_ids": [1, 2],
            "sampling": {
                "max_tokens": 4,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "ignore_eos": False,
                "extra": {
                    "stop": ["<END>"],
                    "include_stop_str_in_output": True,
                    "min_p": 0.05,
                    "thinking_token_budget": 3,
                    "skip_special_tokens": False,
                    "spaces_between_special_tokens": False,
                },
            },
        })))

        self.assertIsInstance(response, FakeStreamingResponse)
        asyncio.run(collect_sse(response))

        self.assertEqual(llm.calls[0]["sampling_params"].kwargs, {
            "max_tokens": 4,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": None,
            "ignore_eos": False,
            "stop": ["<END>"],
            "include_stop_str_in_output": True,
            "min_p": 0.05,
            "thinking_token_budget": 3,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        })

    def test_generate_route_disables_response_model_generation(self):
        # Real FastAPI 0.121.3 raises FastAPIError for the
        # `StreamingResponse | JSONResponse` return annotation unless
        # response_model=None is passed, and it raises it at route-registration
        # time, so the worker dies before it can serve anything (job 314676).
        llm = FakeLLM(FakeRequestOutput([]))
        app = self.worker.create_app(llm, FakePreprocessedInputs, FakeSamplingParams)
        kwargs = app.route_kwargs[("POST", "/generate")]
        self.assertIn("response_model", kwargs)
        self.assertIsNone(kwargs["response_model"])

    def test_generate_accepts_a_request_that_omits_extra(self):
        # The Rust wire type omits `extra` when it is empty, which is the common
        # case; the worker must treat that exactly like `"extra": {}`.
        request_output = FakeRequestOutput([
            FakeOutput(FakeCompletion([31], "ok", "stop")),
        ])
        llm = FakeLLM(request_output)
        app = self.worker.create_app(llm, FakePreprocessedInputs, FakeSamplingParams)
        response = asyncio.run(app.routes[("POST", "/generate")](FakeRequest({
            "request_id": "request-11",
            "prompt_token_ids": [1, 2],
            "sampling": {
                "max_tokens": 4,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "ignore_eos": False,
            },
        })))

        self.assertIsInstance(response, FakeStreamingResponse)
        asyncio.run(collect_sse(response))

        self.assertEqual(llm.calls[0]["sampling_params"].kwargs, {
            "max_tokens": 4,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": None,
            "ignore_eos": False,
        })

    def test_disconnect_aborts_pinned_request_output(self):
        request_output = FakeRequestOutput([
            FakeOutput(FakeCompletion([17], "x", None)),
        ])
        llm = FakeLLM(request_output)
        app = self.worker.create_app(llm, FakePreprocessedInputs, FakeSamplingParams)
        response = asyncio.run(app.routes[("POST", "/generate")](FakeRequest({
            "request_id": "request-9", "prompt_token_ids": [1],
            "sampling": {"max_tokens": 1, "temperature": 0.0, "top_p": 1.0,
                         "top_k": -1, "ignore_eos": False, "extra": {}},
        }, disconnected=True)))

        self.assertEqual(asyncio.run(collect_sse(response)), [])
        self.assertTrue(request_output.aborted)

    def test_generation_error_emits_one_terminal_error_frame(self):
        request_output = FakeRequestOutput([
            FakeOutput(FakeCompletion([], "", None), error="backend failed"),
        ])
        llm = FakeLLM(request_output)
        app = self.worker.create_app(llm, FakePreprocessedInputs, FakeSamplingParams)
        response = asyncio.run(app.routes[("POST", "/generate")](FakeRequest({
            "request_id": "request-11", "prompt_token_ids": [1],
            "sampling": {"max_tokens": 1, "temperature": 0.0, "top_p": 1.0,
                         "top_k": -1, "ignore_eos": False, "extra": {}},
        })))

        self.assertEqual(
            asyncio.run(collect_sse(response)),
            [{"error": "TensorRT-LLM generation failed: backend failed"}],
        )
        self.assertTrue(request_output.aborted)


if __name__ == "__main__":
    unittest.main()
