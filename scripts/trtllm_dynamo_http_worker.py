#!/usr/bin/env python3
"""TensorRT-LLM v1.3.0rc22 HTTP/SSE worker for trtllm-rs Dynamo transport.

Runtime setup (a real TensorRT-LLM runtime is required; this script has no
fallback engine):

    python -m pip install fastapi uvicorn
    export TRTLLM_WORKER_MODEL=/path/to/model-or-engine
    python scripts/trtllm_dynamo_http_worker.py

Environment defaults: ``TRTLLM_WORKER_HOST=127.0.0.1`` and
``TRTLLM_WORKER_PORT=8080``.  Equivalent CLI flags are ``--model``, ``--host``
and ``--port``.  The TensorRT-LLM installation must be the repository-pinned
v1.3.0rc22 runtime and its CUDA/TensorRT prerequisites must already be set up.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import socket
from collections.abc import AsyncIterator, Mapping
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


_REQUEST_FIELDS = frozenset({"request_id", "prompt_token_ids", "sampling"})
_SAMPLING_FIELDS = frozenset({
    "max_tokens", "temperature", "top_p", "top_k", "ignore_eos", "seed", "extra",
})
_EXTRA_FIELDS = frozenset({
    "min_tokens", "stop", "stop_token_ids", "include_stop_str_in_output",
    "min_p", "thinking_token_budget", "repetition_penalty", "presence_penalty",
    "frequency_penalty",
    "skip_special_tokens", "spaces_between_special_tokens",
})
_FINISH_REASONS = frozenset({
    "eos", "stop", "length", "timeout", "cancelled", "content_filter",
})


class ProtocolError(ValueError):
    """A request does not match the narrow TransportRequest wire contract."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def _require_object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError("invalid_request", f"{name} must be an object")
    return value


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError("invalid_request", f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ProtocolError("invalid_request", f"{name} must be at least {minimum}")
    return value


def _require_number(value: Any, name: str, *, minimum: float | None = None,
                    maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError("invalid_request", f"{name} must be a number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ProtocolError("invalid_request", f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ProtocolError("invalid_request", f"{name} must be at most {maximum}")
    return result


def _reject_unknown(mapping: Mapping[str, Any], allowed: frozenset[str], name: str,
                    code: str = "unsupported_request_field") -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ProtocolError(code, f"unsupported {name}: {', '.join(unknown)}")


def _sampling_params(request: Mapping[str, Any], sampling_params_cls: type[Any]) -> Any:
    sampling = _require_object(request.get("sampling"), "sampling")
    _reject_unknown(sampling, _SAMPLING_FIELDS, "sampling field")
    # `extra` is deliberately NOT required. trtllm-core's SamplingParams carries
    # `#[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]` on
    # it, so a request with no backend-specific controls -- the common case --
    # omits the key entirely. Rejecting that would 400 every plain request. The
    # Rust side treats absent and `{}` as the same value, and so does this.
    required = {"max_tokens", "temperature", "top_p", "top_k", "ignore_eos"}
    missing = sorted(required - set(sampling))
    if missing:
        raise ProtocolError("invalid_request", f"sampling is missing: {', '.join(missing)}")

    top_k = _require_int(sampling["top_k"], "sampling.top_k", minimum=-1)
    kwargs: dict[str, Any] = {
        "max_tokens": _require_int(sampling["max_tokens"], "sampling.max_tokens", minimum=1),
        "temperature": _require_number(sampling["temperature"], "sampling.temperature", minimum=0.0),
        "top_p": _require_number(sampling["top_p"], "sampling.top_p", minimum=0.0, maximum=1.0),
        # trtllm-rs uses -1 as its backend-neutral unspecified sentinel.
        "top_k": None if top_k == -1 else top_k,
        "ignore_eos": sampling["ignore_eos"],
    }
    if not isinstance(kwargs["ignore_eos"], bool):
        raise ProtocolError("invalid_request", "sampling.ignore_eos must be a boolean")
    if "seed" in sampling:
        kwargs["seed"] = _require_int(sampling["seed"], "sampling.seed", minimum=0)

    extra = _require_object(sampling.get("extra", {}), "sampling.extra")
    _reject_unknown(extra, _EXTRA_FIELDS, "sampling.extra field", "unsupported_sampling_extra")
    for name in ("min_tokens",):
        if name in extra:
            kwargs[name] = _require_int(extra[name], f"sampling.extra.{name}", minimum=0)
    if "stop" in extra:
        stop = extra["stop"]
        if isinstance(stop, str):
            kwargs["stop"] = stop
        elif isinstance(stop, list) and all(isinstance(value, str) for value in stop):
            kwargs["stop"] = stop
        else:
            raise ProtocolError(
                "invalid_request", "sampling.extra.stop must be a string or list of strings"
            )
    if "stop_token_ids" in extra:
        if not isinstance(extra["stop_token_ids"], list):
            raise ProtocolError("invalid_request", "sampling.extra.stop_token_ids must be a list")
        kwargs["stop_token_ids"] = [
            _require_int(value, "sampling.extra.stop_token_ids[]", minimum=0)
            for value in extra["stop_token_ids"]
        ]
    for name in ("repetition_penalty", "presence_penalty", "frequency_penalty"):
        if name in extra:
            kwargs[name] = _require_number(extra[name], f"sampling.extra.{name}")
    if "min_p" in extra:
        kwargs["min_p"] = _require_number(
            extra["min_p"], "sampling.extra.min_p", minimum=0.0, maximum=1.0
        )
    if "thinking_token_budget" in extra:
        kwargs["thinking_token_budget"] = _require_int(
            extra["thinking_token_budget"], "sampling.extra.thinking_token_budget", minimum=0
        )
    if "include_stop_str_in_output" in extra:
        if not isinstance(extra["include_stop_str_in_output"], bool):
            raise ProtocolError(
                "invalid_request", "sampling.extra.include_stop_str_in_output must be a boolean"
            )
        kwargs["include_stop_str_in_output"] = extra["include_stop_str_in_output"]
    for name in ("skip_special_tokens", "spaces_between_special_tokens"):
        if name in extra:
            if not isinstance(extra[name], bool):
                raise ProtocolError("invalid_request", f"sampling.extra.{name} must be a boolean")
            kwargs[name] = extra[name]
    return sampling_params_cls(**kwargs)


def _parse_request(payload: Any, preprocessed_inputs_cls: type[Any],
                   sampling_params_cls: type[Any]) -> tuple[Any, Any]:
    request = _require_object(payload, "request")
    _reject_unknown(request, _REQUEST_FIELDS, "request field")
    missing = sorted(_REQUEST_FIELDS - set(request))
    if missing:
        raise ProtocolError("invalid_request", f"request is missing: {', '.join(missing)}")
    if not isinstance(request["request_id"], str) or not request["request_id"]:
        raise ProtocolError("invalid_request", "request_id must be a non-empty string")
    prompt_token_ids = request["prompt_token_ids"]
    if not isinstance(prompt_token_ids, list):
        raise ProtocolError("invalid_request", "prompt_token_ids must be a list")
    prompt = [_require_int(token_id, "prompt_token_ids[]", minimum=0)
              for token_id in prompt_token_ids]
    if not prompt:
        raise ProtocolError("invalid_request", "prompt_token_ids must not be empty")
    return (preprocessed_inputs_cls(prompt_token_ids=prompt),
            _sampling_params(request, sampling_params_cls))


def _sse(payload: Mapping[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _abort_if_unfinished(request_output: Any) -> None:
    if not getattr(request_output, "finished", False):
        request_output.abort()


async def _abort_on_disconnect(request: Request, request_output: Any) -> None:
    while not getattr(request_output, "finished", False):
        if await request.is_disconnected():
            _abort_if_unfinished(request_output)
            return
        await asyncio.sleep(0.25)


def create_app(llm: Any, preprocessed_inputs_cls: type[Any],
               sampling_params_cls: type[Any]) -> FastAPI:
    """Build the HTTP app around an already-created pinned TensorRT-LLM LLM."""
    app = FastAPI(title="trtllm-rs Dynamo worker")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # response_model=None is load-bearing, not decoration. FastAPI tries to
    # build a response model from the return annotation, and a union of
    # Response subclasses is not a valid Pydantic field type -- real FastAPI
    # 0.121.3 raises FastAPIError at import time and the worker never starts
    # (job 314676). The annotation is kept because it is true; the response
    # model generation is what has to be turned off.
    @app.post("/generate", response_model=None)
    async def generate(request: Request) -> StreamingResponse | JSONResponse:
        try:
            payload = await request.json()
            inputs, sampling_params = _parse_request(
                payload, preprocessed_inputs_cls, sampling_params_cls)
        except ProtocolError as exc:
            return _error(422, exc.code, str(exc))
        except Exception as exc:
            return _error(400, "invalid_json", str(exc))

        try:
            request_output = llm.generate_async(
                inputs=inputs, sampling_params=sampling_params, streaming=True)
        except Exception as exc:
            return _error(503, "generation_start_failed", str(exc))

        async def events() -> AsyncIterator[str]:
            terminal_sent = False
            disconnect_task = asyncio.create_task(
                _abort_on_disconnect(request, request_output))
            try:
                async for result in request_output:
                    if await request.is_disconnected():
                        _abort_if_unfinished(request_output)
                        return
                    if getattr(result, "error", None):
                        raise RuntimeError(f"TensorRT-LLM generation failed: {result.error}")
                    outputs = getattr(result, "outputs", None)
                    if not isinstance(outputs, list) or len(outputs) != 1:
                        raise RuntimeError("TensorRT-LLM response must contain exactly one output")
                    output = outputs[0]
                    token_ids = list(getattr(output, "token_ids_diff", []))
                    text = getattr(output, "text_diff", "")
                    if not isinstance(text, str):
                        raise RuntimeError("TensorRT-LLM output text_diff must be a string")
                    for index, token_id in enumerate(token_ids):
                        token = _require_int(token_id, "TensorRT-LLM token_id", minimum=0)
                        # A speculative step can contain several token IDs but only one text
                        # delta. Preserve all IDs and attach that delta to its final frame.
                        yield _sse({"token_id": token,
                                    "text": text if index == len(token_ids) - 1 else ""})
                    finish_reason = getattr(output, "finish_reason", None)
                    if finish_reason is not None:
                        if finish_reason not in _FINISH_REASONS:
                            raise RuntimeError(f"unsupported TensorRT-LLM finish_reason: {finish_reason!r}")
                        yield _sse({"finish_reason": finish_reason})
                        terminal_sent = True
                        return
                if not terminal_sent:
                    raise RuntimeError("TensorRT-LLM stream ended without a finish_reason")
            except asyncio.CancelledError:
                _abort_if_unfinished(request_output)
                raise
            except Exception as exc:
                # The Rust transport represents an error SSE frame as a
                # terminal typed error and closes the local stream. A second
                # finish_reason frame would be unreachable after that error.
                yield _sse({"error": str(exc)})
            finally:
                disconnect_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await disconnect_task
                _abort_if_unfinished(request_output)

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def _runtime_types() -> tuple[type[Any], type[Any], type[Any]]:
    try:
        from tensorrt_llm import LLM, SamplingParams
        from tensorrt_llm.llmapi.llm import PreprocessedInputs
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT-LLM v1.3.0rc22 is required; install the pinned runtime "
            "and its CUDA/TensorRT dependencies before starting this worker.") from exc
    return LLM, PreprocessedInputs, SamplingParams


def _engine_kwargs(free_gpu_memory_fraction: float,
                   tensor_parallel_size: int,
                   moe_expert_parallel_size: int | None = None,
                   max_batch_size: int | None = None,
                   max_num_tokens: int | None = None,
                   max_seq_len: int | None = None,
                   enable_chunked_prefill: bool = False) -> dict[str, Any]:
    """Engine settings that must match dynamo.trtllm's, or a comparison lies.

    This worker exists to be measured against `python3 -m dynamo.trtllm`, so any
    LLM() argument that differs between the two shows up as a control-plane
    result when it is really an engine result. dynamo's own construction is
    third_party/dynamo/components/src/dynamo/trtllm/workers/llm_worker.py:257-330.

    Only the arguments that actually differ are set here, and each is here for a
    reason rather than for completeness:

      kv_cache_config       dynamo passes free_gpu_memory_fraction explicitly;
                            leaving it unset gives TensorRT-LLM's 0.9 default,
                            so the two arms sized their KV caches differently
                            (122.81 GiB vs ~109 GiB, job 314683's log).
      tensor_parallel_size  dynamo passes it explicitly; default is already 1,
                            set for symmetry so the value is visible in the log.

    Deliberately NOT set, having been checked against the pinned source:
      scheduler_config      TensorRT-LLM already defaults capacity_scheduler_policy
                            to GUARANTEED_NO_EVICT (llm_args.py:3226-3228), which is
                            what dynamo asks for; and its DynamicBatchConfig is a
                            no-op on the PyTorch backend by TensorRT-LLM's own
                            docstring (llm_args.py:3233-3237).
      return_perf_metrics / enable_iter_perf_stats
                            dynamo derives both from --publish-kv-events, whose
                            default is False (backend_args.py:167-172), so neither
                            arm does per-iteration stats work.
    """
    try:
        from tensorrt_llm.llmapi import KvCacheConfig
    except ImportError as exc:  # pragma: no cover - runtime-only path
        raise RuntimeError("TensorRT-LLM llmapi is required") from exc
    kwargs: dict[str, Any] = {
        "kv_cache_config": KvCacheConfig(
            free_gpu_memory_fraction=free_gpu_memory_fraction),
        "tensor_parallel_size": tensor_parallel_size,
    }
    # Only set when asked. TensorRT-LLM's own defaults are derived from the
    # checkpoint, and passing None would override that derivation with nothing.
    if enable_chunked_prefill:
        # OFF by default in TensorRT-LLM rc22 (llm_args.py:4115), and that
        # default makes the competition's 20 ms ITL gate unreachable at ISL
        # 4000: an unchunked 4000-token prefill is one forward pass that stalls
        # every generating sequence in the same iteration. Job 314882 measured
        # ~92 ms per iteration with a decode batch of ONE, which is the floor,
        # not a scheduling choice. `--cache-bust first_turn_prefix` means every
        # request pays that prefill in full, by design.
        kwargs["enable_chunked_prefill"] = True
    for name, value in (
        ("moe_expert_parallel_size", moe_expert_parallel_size),
        ("max_batch_size", max_batch_size),
        ("max_num_tokens", max_num_tokens),
        ("max_seq_len", max_seq_len),
    ):
        if value is not None:
            kwargs[name] = value
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("TRTLLM_WORKER_MODEL"),
                        help="model or engine path (or TRTLLM_WORKER_MODEL)")
    parser.add_argument("--host", default=os.environ.get("TRTLLM_WORKER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("TRTLLM_WORKER_PORT", "8080")))
    # Default 0.80, not TensorRT-LLM's 0.9: this worker's whole purpose is to be
    # compared against dynamo.trtllm, which is always launched with 0.80 here.
    parser.add_argument("--free-gpu-memory-fraction", type=float,
                        default=float(os.environ.get(
                            "TRTLLM_WORKER_FREE_GPU_MEMORY_FRACTION", "0.80")))
    parser.add_argument("--tensor-parallel-size", type=int,
                        default=int(os.environ.get("TRTLLM_WORKER_TP", "1")))
    # MoE and shape knobs, needed for Qwen3-235B-A22B. Values come from NVIDIA's
    # own H200 recipe (recipes/qwen3-235b-a22b-fp8/trtllm/agg/hopper/deploy.yaml:
    # tensor_parallel_size 4, moe_expert_parallel_size 4, max-batch-size 128,
    # max-num-tokens 8192, max-seq-len 8192) rather than being guessed here.
    parser.add_argument("--moe-expert-parallel-size", type=int, default=None)
    parser.add_argument("--max-batch-size", type=int, default=None)
    parser.add_argument("--max-num-tokens", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--enable-chunked-prefill", action="store_true",
                        help="chunk prefill so it stops stalling decode; see _engine_kwargs")
    args = parser.parse_args()
    if not args.model:
        parser.error("--model or TRTLLM_WORKER_MODEL is required")
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("FastAPI runtime requires uvicorn; install it before starting the worker.") from exc
    llm_cls, preprocessed_inputs_cls, sampling_params_cls = _runtime_types()
    engine_kwargs = _engine_kwargs(
        args.free_gpu_memory_fraction,
        args.tensor_parallel_size,
        args.moe_expert_parallel_size,
        args.max_batch_size,
        args.max_num_tokens,
        args.max_seq_len,
        args.enable_chunked_prefill,
    )
    printable = {
        k: (v if not hasattr(v, "free_gpu_memory_fraction") else
            f"KvCacheConfig(free_gpu_memory_fraction={v.free_gpu_memory_fraction})")
        for k, v in engine_kwargs.items()
    }
    print(f"engine config: {printable}", flush=True)
    app = create_app(llm_cls(model=args.model, **engine_kwargs),
                     preprocessed_inputs_cls, sampling_params_cls)
    # uvicorn 0.51.0 never sets TCP_NODELAY -- grep its source, there is not one
    # occurrence -- so its SSE frames go out under Nagle. One token per frame is
    # exactly the traffic Nagle punishes, and with the peer's delayed ACK it
    # costs ~40 ms per token (job 315393: engine 14.16 ms/token, AIPerf mean ITL
    # 39.06 ms with a 13.95 ms minimum).
    #
    # Set on the *listening* socket: on Linux accepted sockets inherit
    # TCP_NODELAY, so this covers every connection without patching uvicorn.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.bind((args.host, args.port))
    sock.listen(2048)
    print(f"listening on {args.host}:{args.port} with TCP_NODELAY", flush=True)
    server = uvicorn.Server(uvicorn.Config(app, log_level="info"))
    server.run(sockets=[sock])


if __name__ == "__main__":
    main()
