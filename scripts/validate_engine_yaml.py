"""Reject an engine.yaml TensorRT-LLM will not do what you think with.

Four ways a setting can be wrong and produce no error at the point you write it:

  0. A combination each field accepts and the engine as a whole does not. The
     argument object is built here, so cross-field validators run.
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
from pydantic import ValidationError
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# Fields typed as plain `str` accept anything; these are the values that mean
# something. attention_backend/{trtllm,flashinfer,triton_prefill,vanilla}.py
ALLOWED = {
    "attn_backend": ("TRTLLM", "FLASHINFER", "TRITON", "VANILLA"),
}

# Choices that type-check and then fail at import.
#
# The module names matter and one of them was wrong here: DEEPGEMM does NOT
# need the PyPI `deep_gemm` package. TensorRT-LLM bundles its own
# `tensorrt_llm.deep_gemm`, which is what fused_moe_deepgemm.py imports, and it
# is present with every symbol that path uses. Gating on the top-level name
# rejected a backend that works -- the same class of error as reading a
# declared default instead of the branch that consumes it.
# A second kind of gate, and the distinction is the whole point: GATED asks
# "is the library installed", SM_GATED asks "does this GPU run these kernels".
# They are different questions and this file conflated them until DEEPGEMM was
# recommended for an H200. `import tensorrt_llm.deep_gemm` succeeds there --
# the module ships regardless -- and DeepGemmFusedMoE then refuses at
# construction with "requires SM100 or SM103" (fused_moe_deepgemm.py:751),
# inside the GPU allocation, which is the expensive place to find out.
#
# Values are (predicate on sm, explanation). Cited by file:line so a reader can
# check the claim instead of trusting the table.
SM_GATED = {
    ("moe_config", "backend", "DEEPGEMM"): (
        lambda sm: sm in (100, 103),
        "DeepGemmFusedMoE requires SM100 or SM103 (fused_moe_deepgemm.py:751). "
        "vLLM's deep_gemm is a different implementation; a result from that "
        "stack does not transfer to this flag.",
    ),
    ("moe_config", "backend", "TRTLLM"): (
        lambda sm: sm in (100, 103),
        "TRTLLMGenFusedMoE requires SM100 or SM103 (fused_moe_trtllm_gen.py:157).",
    ),
    ("moe_config", "backend", "MARLIN"): (
        lambda sm: sm == 90,
        "MarlinFusedMoE is SM90-only and additionally requires an NVFP4 "
        "checkpoint (fused_moe_marlin.py:73,78).",
    ),
    ("moe_config", "backend", "TRITON"): (
        lambda sm: sm == 90,
        "TritonFusedMoE is SM90-only (fused_moe_triton.py:1499).",
    ),
}


def _sm_version():
    """Compute capability x10, or None when there is no GPU to ask.

    None means "not checked", never "supported": a validator that passes on a
    login node because it cannot see a GPU would be worse than one that does
    not check at all.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability()
        return major * 10 + minor
    except Exception:
        return None


GATED = {
    ("moe_config", "backend", "DEEPGEMM"): "tensorrt_llm.deep_gemm",
    ("moe_config", "backend", "MEGAMOE_DEEPGEMM"): "tensorrt_llm.deep_gemm",
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

    # Build the real argument object rather than checking fields one at a time.
    # TorchLlmArgs requires only `model`, and constructing it runs the
    # cross-field validators too -- which per-field TypeAdapter checks cannot
    # reach. `cuda_graph_config` resolving to DecodeCudaGraphConfig with the
    # values we asked for is a stronger statement than "the annotation accepts
    # this dict".
    known_doc = {k: v for k, v in doc.items() if k not in unknown}
    if known_doc:
        try:
            TorchLlmArgs(model="/dev/null", **known_doc)
        except ValidationError as exc:
            for err in exc.errors():
                loc = ".".join(str(x) for x in err["loc"]) or "?"
                problems.append(f"{loc} rejected -- {err['msg']}")
        except Exception as exc:  # a validator that raises something else
            problems.append(f"engine args rejected -- {type(exc).__name__}: {exc}")

    for key, allowed in ALLOWED.items():
        chosen = doc.get(key)
        if chosen is not None and chosen not in allowed:
            problems.append(
                f"{key}={chosen!r} is not one of {', '.join(allowed)} "
                f"(the field is typed `str`, so nothing else catches this)")

    sm = _sm_version()
    for (key, sub, want), (ok, why) in SM_GATED.items():
        chosen = doc.get(key) if sub is None else (doc.get(key) or {}).get(sub)
        if chosen != want:
            continue
        where = key if sub is None else f"{key}.{sub}"
        if sm is None:
            problems.append(
                f"{where} = {want!r}: no GPU visible here, so this was NOT "
                f"checked. On the wrong hardware it fails at construction. {why}"
            )
        elif not ok(sm):
            problems.append(f"{where} = {want!r} on SM{sm}: {why}")

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
