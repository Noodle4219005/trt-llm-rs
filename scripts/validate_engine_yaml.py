"""Reject engine.yaml keys TensorRT-LLM will silently ignore.

`update_llm_args_with_extra_options` merges the YAML into the argument dict
without checking anything: a file containing `cuda_graph_confgi` and
`totally_not_a_real_option: 7` is accepted verbatim. Combined with
`--enable-cuda-graph`, which `dynamo.trtllm` accepts and wires only into its
diffusion engine, that makes two ways to configure something and have nothing
happen -- job 316849 passed the flag and logged `cuda_graph=False`.

Ten seconds here, before 220 GiB of weights are loaded.
"""

import sys

import yaml
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

path = sys.argv[1]
with open(path) as handle:
    doc = yaml.safe_load(handle) or {}

known = set(TorchLlmArgs.model_fields)
unknown = sorted(k for k in doc if k not in known)
if unknown:
    print(f"!!! {path}: TensorRT-LLM does not have these options, and it will "
          f"NOT tell you: {', '.join(unknown)}", file=sys.stderr)
    for k in unknown:
        near = sorted(f for f in known if f[:4] == k[:4])
        if near:
            print(f"!!!   {k} -> did you mean {', '.join(near)}?", file=sys.stderr)
    raise SystemExit(1)

print(f"    engine.yaml: {len(doc)} option(s) accepted -- {', '.join(sorted(doc))}")
