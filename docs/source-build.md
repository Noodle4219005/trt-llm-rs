# Source-build inspection gate

This repository's source-build baseline is the following pair of initialized,
clean Git submodules:

| Source | Required release | Required commit | Directory |
| --- | --- | --- | --- |
| NVIDIA Dynamo | v1.4.1 | `2112d6ba74da72e2715ae69f4b76458b7691380d` | `third_party/dynamo` |
| NVIDIA TensorRT-LLM | v1.3.0rc22 | `8ba93401976877ca2a390104829dd0d54cf2f30f` | `third_party/TensorRT-LLM` |

Run the inspection gate before configuring or compiling source dependencies:

```bash
./scripts/verify-source-tree.sh
```

The gate is deliberately read-only. It fails closed when either directory is
missing or is not a Git worktree, when the superproject Gitlink or checked-out
submodule `HEAD` differs from the required commit, or when either submodule
has tracked, staged, or untracked changes. It does not initialize, fetch,
update, reset, clean, or modify submodules.

## Initial source checkout

For a new checkout, initialize the exact source directories before using the
gate:

```bash
git clone --recurse-submodules <repository-url> trt-llm-rs
cd trt-llm-rs
git submodule update --init --recursive
./scripts/verify-source-tree.sh
```

If a repository was cloned without submodules, run only the
`git submodule update --init --recursive` command from its root, then rerun the
gate. The gate never performs that initialization itself.

## Compilation prerequisites and commands

The following are prerequisites to arrange before a from-source compilation:

- A supported Linux host with Git, Bash, CMake, and a C/C++ compiler toolchain.
- A CUDA-capable NVIDIA environment, a CUDA toolkit compatible with the chosen
  TensorRT-LLM release, and the TensorRT libraries/development headers required
  by that release.
- Python 3 with an isolated virtual environment plus the Python build tooling
  required by the selected Dynamo and TensorRT-LLM source-build instructions.
- Rust and Cargo for the `trt-llm-rs` workspace itself.

After satisfying the upstream release-specific prerequisites, the intended
command sequence is:

```bash
./scripts/verify-source-tree.sh
# Follow third_party/dynamo's v1.4.1 source-build instructions.
# Follow third_party/TensorRT-LLM's v1.3.0rc22 source-build instructions.
cargo build --workspace
```

These are documented commands only; this gate does not claim that any
compilation, installation, or upstream build command has been run.

## Source-visible runtime wiring

The runtime replacement is split into a pinned TensorRT-LLM Python worker and
the Rust Dynamo v1.4.1 worker. The Python process owns the CUDA/TensorRT-LLM
engine; the Rust process owns the Dynamo Worker lifecycle and sends requests
over the concrete HTTP/SSE transport. There is no fake or CPU fallback.

Start the Python worker after its source-build environment is active:

```bash
export TRTLLM_WORKER_MODEL=/path/to/model-or-engine
python scripts/trtllm_dynamo_http_worker.py \
  --host 127.0.0.1 \
  --port 8080
```

In a second process, run the Rust Dynamo entrypoint from the nested crate:

```bash
export TRTLLM_WORKER_URL=http://127.0.0.1:8080
export TRTLLM_MODEL=/path/to/model-or-engine
cargo run --manifest-path crates/dynamo/Cargo.toml \
  --features dynamo-v1 \
  --example dynamo-worker
```

The Python worker exposes GET /health and POST /generate. Rust cancellation
drops the local SSE response stream; it does not call an unimplemented remote
cancel endpoint. Run ./scripts/verify-source-tree.sh before both source-build
and runtime gates, and retain the two upstream source directories for
competition inspection.

## Archive limitation

A shallow clone can contain the pinned commits and pass this inspection gate,
but it is not an archival full-history bundle. Preserve a non-shallow clone or
an explicit Git bundle when full upstream history is required for archival,
provenance, or later history-dependent inspection.
