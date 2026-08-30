"""
trtllm_rs_bridge: swaps TensorRT-LLM PyExecutor's scheduler policy object for
a Rust-backed one, without touching any other mechanism (speculative
decoding, CUDA graphs, sampling, KV, disaggregation all stay upstream's).

See `scheduler.RustBackedCapacityScheduler` for the swap itself, `patch.install` /
`patch.install_global` for how to wire it into a running or about-to-be-built
PyExecutor, and `_abi` for the flatten/decide/unflatten crossing into Rust.
"""

from ._abi import flatten, unflatten
from .patch import InstallHandle, install, install_global, wrapped_executors
from .scheduler import Divergence, RustBackedCapacityScheduler

__all__ = [
    "RustBackedCapacityScheduler",
    "Divergence",
    "install",
    "install_global",
    "wrapped_executors",
    "InstallHandle",
    "flatten",
    "unflatten",
]
