"""Load safetensors shards as mmap views so the ranks on a node share one copy.

TensorRT-LLM's `HfWeightLoader._load_safetensors_file` is one line:

    return safetensors.torch.load_file(file)

`load_file` mmaps the shard and then *copies* every tensor out of it into
anonymous memory. Anonymous memory is private to a process, so eight TP ranks
on one node each hold their own copy of the same checkpoint. For
`Qwen3-235B-A22B-FP8` that is 221 GB per rank:

    TP4:  4 x 221 GB =  884 GB  <  1600 GB   loads, every time
    TP8:  8 x 221 GB = 1.77 TB  >  1600 GB   fails, every time

1600 GB is nano4's hard ceiling -- the quota is 200 GB per GPU and an 8-GPU
node cannot be given more. The failure surfaces as
`unable to mmap ~10 GB: Cannot allocate memory (12)` followed by
`can't start new thread`, which reads like a thread-limit problem and is not.

Returning read-only views keeps the pages file-backed, so the eight ranks share
one page-cache copy per node instead of holding eight private ones.

The mapping must be MAP_SHARED|PROT_READ (`mode="r"`), not copy-on-write
(`mode="c"`). This node runs `vm.overcommit_memory=2` -- strict accounting --
with a node-wide CommitLimit of 1603 GiB and no cgroup memory.max at all. A
writable MAP_PRIVATE mapping reserves every page it could dirty, so the two
modes differ by the entire checkpoint (measured, job 316262):

    mode='r':  Committed_AS 93.8 -> 93.8  GiB   (+0.0 for 220.2 GiB mapped)
    mode='c':  Committed_AS 93.8 -> 314.0 GiB   (+220.2)

Eight ranks x 220.2 GiB of copy-on-write charge is 1762 GiB against a 1603 GiB
limit, which is why job 316229 still died at ENOMEM -- inside our own np.memmap
this time -- after roughly 20 of 24 shards. Read-only charges nothing, so the
same eight ranks charge nothing.

The tensors are therefore read-only. torch accepts them (job 316262: a
UserWarning, then .view() and .cuda() both correct), but a WRITE to one is a
SIGSEGV rather than an exception. That is the standing risk of this approach:
weights are semantically read-only and TensorRT-LLM copies out of them, but
nothing enforces it.

This replaces one static method and changes no upstream file. Enable with
TRTLLM_RS_MMAP_WEIGHTS=1.
"""

import json
import struct
import sys
import warnings
from typing import Any, Dict, List

import numpy as np
import torch


def _dtypes() -> Dict[str, Any]:
    # torch.uint16/32/64 arrived in 2.3; a checkpoint that uses them on an
    # older torch should say so rather than silently mis-reinterpret bytes.
    table = {
        "BOOL": torch.bool,
        "U8": torch.uint8,
        "I8": torch.int8,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E5M2": torch.float8_e5m2,
        "I16": torch.int16,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I32": torch.int32,
        "F32": torch.float32,
        "I64": torch.int64,
        "F64": torch.float64,
    }
    for name, attr in (("U16", "uint16"), ("U32", "uint32"), ("U64", "uint64")):
        if hasattr(torch, attr):
            table[name] = getattr(torch, attr)
    return table


_DTYPE = None


def _as_tensor(raw: np.ndarray, dtype, shape: List[int]) -> torch.Tensor:
    tensor = torch.from_numpy(raw)
    try:
        tensor = tensor.view(dtype)
    except RuntimeError:
        # A tensor whose byte range is not divisible by its element size, or
        # whose offset torch refuses to reinterpret. Copy this one rather than
        # dropping the sharing for the whole shard.
        tensor = torch.from_numpy(np.array(raw)).view(dtype)
    return tensor.reshape(shape)


def load_file_mmap(file: str) -> Dict[str, torch.Tensor]:
    """`safetensors.torch.load_file` without the copy."""
    global _DTYPE
    if _DTYPE is None:
        _DTYPE = _dtypes()

    with open(file, "rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len))
    data_start = 8 + header_len

    backing = np.memmap(file, dtype=np.uint8, mode="r")

    weights = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype = _DTYPE.get(meta["dtype"])
        if dtype is None:
            raise ValueError(
                f"unsupported safetensors dtype {meta['dtype']!r} for {name!r} in {file}")
        begin, end = meta["data_offsets"]
        weights[name] = _as_tensor(
            backing[data_start + begin:data_start + end], dtype, meta["shape"])
    return weights


def install() -> None:
    """Replace `safetensors.torch.load_file`, not TensorRT-LLM's caller.

    The obvious target is `HfWeightLoader._load_safetensors_file`, but
    `checkpoints/__init__.py:19` does `from .hf.weight_loader import
    HfWeightLoader`: resolving the leaf module imports the parent package,
    which imports the leaf, so by the time a post-import hook on the leaf could
    run the module body has already executed and the patch is lost.
    `safetensors.torch` has no such parent -- `safetensors/__init__.py` does
    not import it -- so hooking it is a single well-defined interception.
    """
    import safetensors.torch as st

    if getattr(st, "_trtllm_rs_mmap", False):
        return
    # "The given NumPy array is not writable" fires once per tensor -- about a
    # thousand per shard, per rank. Filter it once here rather than wrapping
    # each conversion: warnings.catch_warnings mutates global filter state and
    # is not thread-safe, and this runs on 32 loader threads.
    warnings.filterwarnings("ignore", message=".*not writable.*",
                            category=UserWarning)
    original = st.load_file

    def load_file(filename, device="cpu"):
        # Only the CPU path can be a mapping. Anything asking for device
        # memory gets upstream's implementation unchanged.
        if str(device) != "cpu":
            return original(filename, device=device)
        return load_file_mmap(str(filename))

    st.load_file = load_file
    st._trtllm_rs_mmap = True
    print("[trtllm-rs-mmap] safetensors shards load as mmap views",
          file=sys.stderr, flush=True)
