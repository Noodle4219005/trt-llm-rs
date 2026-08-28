#!/usr/bin/env bash
# Resolve and download every crate, on the login node.
#
# Compute nodes on this cluster may have no route to crates.io, and a build that
# discovers that after allocating GPUs has wasted the allocation. `cargo fetch`
# is pure network I/O with no compilation, so it belongs here and not in a job.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo fetch --locked 2>/dev/null || cargo fetch
echo "vendored into ${CARGO_HOME:-$HOME/.cargo}/registry; build with --offline"
