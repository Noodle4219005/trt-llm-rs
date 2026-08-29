# Patches against pinned upstream

Patches here apply to `third_party/` at the SHAs recorded in `../VERSIONS.md`.
Each file is a `git format-patch` output with a header explaining **why** the
change is necessary — not what it does, which the diff already says.

## The rule

**Prefer not to patch.** Our replacement for TensorRT-LLM's Python control plane
lives in `crates/pyengine/python/trtllm_rs_bridge/`, which substitutes behaviour
without touching upstream. A patch is only justified when upstream offers no
seam — for example `TrtllmAttentionMetadata.prepare` calls
`kv_cache_manager.impl.copy_batch_block_offsets`, a pybind method on a C++
object that cannot be overridden from Python (see the C4 stage of the plan).

Every patch is a standing liability: it must be re-verified on every upstream
version bump. Name that cost in the patch header, so whoever bumps the version
knows what they are signing up for.
