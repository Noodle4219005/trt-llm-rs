#!/usr/bin/env bash
# Verify the pinned source trees required for a from-source build.
# This script is intentionally read-only: it never fetches, initializes,
# updates, resets, cleans, or otherwise changes Git state.

set -euo pipefail

readonly repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

fail() {
    printf 'source-tree verification failed: %s\n' "$*" >&2
    exit 1
}

require_clean_pinned_submodule() {
    local path="$1"
    local expected_commit="$2"
    local source_dir="${repo_root}/${path}"
    local gitlink_commit
    local head_commit
    local worktree_status

    [[ -d "${source_dir}" ]] || fail "missing initialized submodule directory: ${path}"
    git -C "${source_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
        || fail "submodule is not a Git worktree: ${path}"

    gitlink_commit="$(git -C "${repo_root}" ls-tree HEAD -- "${path}" | awk '{print $3}')"
    [[ "${gitlink_commit}" == "${expected_commit}" ]] \
        || fail "superproject gitlink mismatch for ${path}: expected ${expected_commit}, got ${gitlink_commit:-missing}"

    head_commit="$(git -C "${source_dir}" rev-parse HEAD)"
    [[ "${head_commit}" == "${expected_commit}" ]] \
        || fail "checked-out commit mismatch for ${path}: expected ${expected_commit}, got ${head_commit}"

    worktree_status="$(git -C "${source_dir}" status --porcelain=v1 --untracked-files=all)"
    [[ -z "${worktree_status}" ]] \
        || fail "submodule worktree is not clean: ${path}"

    printf 'verified %s at %s\n' "${path}" "${expected_commit}"
}

git -C "${repo_root}" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || fail "repository root is not a Git worktree: ${repo_root}"

require_clean_pinned_submodule \
    "third_party/dynamo" \
    "2112d6ba74da72e2715ae69f4b76458b7691380d"
require_clean_pinned_submodule \
    "third_party/TensorRT-LLM" \
    "8ba93401976877ca2a390104829dd0d54cf2f30f"

printf 'source-tree verification passed\n'
