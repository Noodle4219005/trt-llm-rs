# Pinned stack

Every version here is exact and must not drift. The pins come from NVIDIA's own
compatibility matrix (`third_party/dynamo/docs/fern/pages/reference/general/compatibility.mdx`),
which maps Dynamo `v1.4.1` to TensorRT-LLM `1.3.0rc22`.

| Component | Version | Commit |
|---|---|---|
| `third_party/dynamo` | **v1.4.1** | `2112d6ba74da72e2715ae69f4b76458b7691380d` |
| `third_party/TensorRT-LLM` | **v1.3.0rc22** | `8ba93401976877ca2a390104829dd0d54cf2f30f` |

Not `rc23` or `rc24`. Dynamo HEAD pins `rc24`, but `v1.4.1` — the version whose
Rust `LLMEngine` trait we implement — pins `rc22`. Anything newer belongs on a
separate experimental branch, not here.

`v1.4.1`'s `LLMEngine` interface (`lib/backend-common/src/engine.rs:172`) is
treated as a **pinned API**. Note the line number: it is 190 on Dynamo HEAD and
172 on v1.4.1, which is the sort of drift that makes an unpinned citation useless.

## Everything is built from source

The deliverable is built from the source in this repository. A prebuilt
`tensorrtllm-runtime` container exists on this cluster and is used for exactly
two things, neither of which is the deliverable:

1. **A build environment.** `apptainer exec <sif> cmake ...` borrows a toolchain
   (nvcc 13.2, cmake 4.0.3, ninja, g++ 13.3, ccache, TensorRT headers) the way
   any build borrows a compiler. The artefact is ours.
2. **A reference baseline.** Phase A compares our stack against the official one.

Using it as a *runtime* would make the performance claim unfalsifiable — nobody
could tell our optimisation from NVIDIA's build flags. That is why it is not a
dependency of the final artefact.

## Layout

```
third_party/          submodules, never modified, pinned to the SHAs above
patches/              our changes to upstream, as reviewable patch files
crates/               our Rust
crates/pyengine/python/trtllm_rs_bridge/   our replacement for TRT-LLM's Python control plane
```

Upstream is never edited in place. Changes live either in `patches/` (when a
line in upstream genuinely has to move) or in `trtllm_rs_bridge/` (the normal
case — we replace behaviour rather than editing it). Both survive an upstream
version bump as a legible diff instead of a merge conflict.
