# KV transfer

`crates/transfer`. The `nixl` feature is a placeholder; the reshard mapping it
depends on is implemented and tested, because that is the half that fails
without an error message.

## The two failures this project has already paid for

**A transfer that moves 0 bytes looks like a slow decode worker.** Every
implementation reports `TransferStats::bytes`, and `moved_data()` must be
checked before anything downstream is interpreted. If it is zero, the run has no
result — not a bad one.

**Heterogeneous TP needs integer division, not modulo.** With `H` KV heads over
`T` ranks: when `T ≤ H` each rank owns `H/T` heads; when `T > H` each head is
*replicated* across `T/H` ranks, and head `h` lands on ranks
`h·(T/H) … h·(T/H)+(T/H−1)`. Writing `h % T` compiles, moves the right number of
bytes, and produces wrong attention output.

Qwen3-235B-A22B has **4 KV heads**. A TP8 decode worker replicates every head
across 2 ranks, so the 4P1D topology (TP2 prefill → TP8 decode) hits this on
every single request. `Reshard::plan()` makes the mapping a value that unit tests
can check without a GPU.

## Sizing

`bytes_per_token = 2 × layers × kv_heads × head_dim × dtype_bytes`
= `2 × 94 × 4 × 128 × 1` = **96 KiB/token** at FP8.

A 4000-token prompt is **367 MiB** per request. At 4P1D and ~18 req/s that is
~6.4 GiB/s of steady KV traffic into one decode worker — comfortably inside
NVLink, and the reason the decode worker wants to be on one node.

## Layer-wise streaming

The prefill worker samples the first token itself, so **TTFT stops at prefill
completion** and the handoff lands inside the inter-token budget instead. Over
199 gaps a 10 ms transfer costs 0.05 ms of mean ITL — negligible. This is why
the transfer path is not on the critical path of the metric, and why optimising
it before the prefill queue is optimising the wrong thing.

Streaming KV per layer as it is produced overlaps the transfer with prefill
compute entirely. Worth doing, worth measuring, not worth doing first.
