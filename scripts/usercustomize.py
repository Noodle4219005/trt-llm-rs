"""Arm the Rust shadow scheduler in every Python process, spawned ones included.

TensorRT-LLM's executor runs in an MPI_Comm_spawn'd interpreter, which does not
inherit a monkeypatch applied in the launching process. `usercustomize` is
imported by site.py in every interpreter with user site enabled, so putting this
directory on PYTHONPATH reaches the executor too.

`usercustomize` and not `sitecustomize`: the container already ships
/usr/lib/python3.12/sitecustomize.py, and shadowing someone else's module to
install our own hook would be a poor trade. usercustomize is unused there.

The patch is applied *after* `_util` finishes importing, never before. Importing
tensorrt_llm from site initialisation would drag a multi-second, import-cycle-prone
dependency into every interpreter on the node -- including ones that have nothing
to do with the engine. The meta-path shim below costs nothing until the one module
we care about is actually imported.

Two independent patches ride this mechanism, each behind its own switch:
TRTLLM_RS_SHADOW=1 installs the Rust scheduler, TRTLLM_RS_MMAP_WEIGHTS=1
installs the mmap safetensors loader. Neither implies the other.
"""

import os
import sys

_SHADOW_TARGET = "tensorrt_llm._torch.pyexecutor._util"
_MMAP_TARGET = "safetensors.torch"


def _arm(target_name: str, installer: str, tag: str) -> None:
    import importlib.abc
    import importlib.util

    class _PatchAfterImport(importlib.abc.MetaPathFinder):
        """Let the normal finders resolve the module, then post-process it."""

        def find_spec(self, fullname, path=None, target=None):
            if fullname != target_name:
                return None
            # Step aside so the real finders can resolve it, then wrap the
            # loader so our patch runs immediately after its module body.
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None

            loader = spec.loader
            original_exec = loader.exec_module

            def exec_module(module):
                original_exec(module)
                try:
                    sys.meta_path.remove(self)
                except ValueError:
                    pass
                try:
                    __import__(installer).install()
                except Exception as exc:
                    print(f"[{tag}] install failed: {exc}",
                          file=sys.stderr, flush=True)

            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _PatchAfterImport())


for _flag, _target, _installer, _tag in (
    ("TRTLLM_RS_SHADOW", _SHADOW_TARGET, "trtllm_rs_shadow", "trtllm-rs-shadow"),
    ("TRTLLM_RS_MMAP_WEIGHTS", _MMAP_TARGET, "trtllm_rs_mmap_weights",
     "trtllm-rs-mmap"),
):
    if os.environ.get(_flag) != "1":
        continue
    try:
        _arm(_target, _installer, _tag)
    except Exception as exc:  # never break an interpreter over this
        print(f"[{_tag}] usercustomize failed: {exc}",
              file=sys.stderr, flush=True)
