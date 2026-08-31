"""Reject an engine.yaml TensorRT-LLM will not do what you think with.

Four ways a setting can be wrong and produce no error at the point you write it:

  1. A misspelt key. `update_llm_args_with_extra_options` merges the YAML into
     the argument dict without checking anything, so `cuda_graph_confgi` and
     `totally_not_a_real_option: 7` are both accepted verbatim.
  2. A value of the wrong shape. cuda_graph_config is a union discriminated on
     `mode`, so a dict without it fails validation -- and CudaGraphConfig.mode
     has a default, which is exactly what makes omitting it look safe.
  3. A misspelt value in a field typed `str`. attn_backend is one, so
     `FLASHINFR` type-checks and silently means "not FlashInfer".
  4. A correct value with no library behind it. DEEPGEMM needs deep_gemm and
     MOONCAKE needs mooncake; neither is importable in this container, so the
     failure lands several minutes and 220 GiB of weights into a run.

Ten seconds here, before the workers start.
"""

import sys

import yaml
from pydantic import TypeAdapter, ValidationError
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# Fields typed as plain `str` accept anything; these are the values that mean
# something. attention_backend/{trtllm,flashinfer,triton_prefill,vanilla}.py
ALLOWED = {
    "attn_backend": ("TRTLLM", "FLASHINFER", "TRITON", "VANILLA"),
}

# Choices that type-check and then fail at import.
GATED = {
    ("moe_config", "backend", "DEEPGEMM"): "deep_gemm",
    ("moe_config", "backend", "MEGAMOE_DEEPGEMM"): "deep_gemm",
    ("moe_config", "backend", "WIDEEP"): "deep_ep",
    ("cache_transceiver_config", "backend", "MOONCAKE"): "mooncake",
    ("attn_backend", None, "FLASHINFER"): "flashinfer",
}


def check(doc: dict) -> list[str]:
    """Every reason TensorRT-LLM would not do what this document says."""
    fields = TorchLlmArgs.model_fields
    known = set(fields)
    problems: list[str] = []

    unknown = sorted(k for k in doc if k not in known)
    for key in unknown:
        near = sorted(f for f in known if f[:4] == key[:4])
        hint = f" -- did you mean {', '.join(near)}?" if near else ""
        problems.append(f"{key}: not an option TensorRT-LLM has{hint}")

    for key, value in doc.items():
        if key in unknown:
            continue
        try:
            TypeAdapter(fields[key].annotation).validate_python(value)
        except ValidationError as exc:
            first = exc.errors()[0]
            loc = ".".join(str(x) for x in first["loc"]) or key
            problems.append(f"{key}: {loc} rejected -- {first['msg']}")

    for key, allowed in ALLOWED.items():
        chosen = doc.get(key)
        if chosen is not None and chosen not in allowed:
            problems.append(
                f"{key}={chosen!r} is not one of {', '.join(allowed)} "
                f"(the field is typed `str`, so nothing else catches this)")

    for (key, sub, want), module in GATED.items():
        chosen = doc.get(key) if sub is None else (doc.get(key) or {}).get(sub)
        if chosen != want:
            continue
        try:
            __import__(module)
        except ImportError:
            where = key if sub is None else f"{key}.{sub}"
            problems.append(
                f"{where}={want} needs the `{module}` module, which is not "
                f"importable here")

    return problems


if __name__ == "__main__":
    path = sys.argv[1]
    with open(path) as handle:
        parsed = yaml.safe_load(handle) or {}
    found = check(parsed)
    if found:
        print(f"!!! {path}:", file=sys.stderr)
        for line in found:
            print(f"!!!   {line}", file=sys.stderr)
        raise SystemExit(1)
    print(f"    engine.yaml: {len(parsed)} option(s) accepted -- "
          f"{', '.join(sorted(parsed))}")
