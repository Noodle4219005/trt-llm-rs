//! Scheduling.
//!
//! `goodput = req/s x good_frac`, and the two factors are pinned by different
//! halves of the deployment:
//!
//! * **`req/s` is pinned by decode.** A good request must emit `osl` tokens
//!   averaging no more than `itl_ms` apart, so it holds a decode slot for at
//!   least `osl * itl_ms`. At OSL 200 and a 20 ms budget that is 4.0 seconds,
//!   and the decode pool cannot exceed `concurrency / 4.0` requests per second
//!   however fast its kernels are.
//! * **`good_frac` is pinned by prefill queueing.** Sweeping concurrency on a
//!   fixed topology, ITL never came within 30 % of its budget while TTFT grew
//!   10.8x and crossed 3000 ms. Requests are lost to *waiting*, not to
//!   arithmetic.
//!
//! So the two schedulers here optimise different things on purpose.
//! [`prefill`] maximises the number of requests that meet their first-token
//! deadline, and [`decode`] maximises sustained concurrency subject to the ITL
//! budget. Neither tries to minimise latency, which is what a stock serving
//! scheduler does and is not what this metric rewards.

pub mod decode;
pub mod policy;
pub mod prefill;

pub use decode::{AdmitDecision, DecodeScheduler, ItlController, RunningSeq};
pub use policy::{order_jobs, Job, JobOrdering};
pub use prefill::{PendingPrefill, PrefillBatch, PrefillChunk, PrefillScheduler};
