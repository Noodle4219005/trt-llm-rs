//! Mapping KV heads from a prefill worker's tensor-parallel layout to a decode
//! worker's.
//!
//! With `H` key/value heads sharded over `T` ranks:
//!
//! * when `T <= H`, each rank owns `H / T` distinct heads;
//! * when `T > H`, there are not enough heads to go round, so each head is
//!   **replicated** across `T / H` ranks.
//!
//! The destination rank for head `h` in the replicated case is therefore
//! `h * (T / H) + r` for `r` in `0 .. T/H`, which is an integer *division*
//! relationship. Writing it as `h % T` compiles, runs, moves the right number
//! of bytes, and produces wrong attention outputs - the failure mode with no
//! error message.
//!
//! Qwen3-235B-A22B has 4 KV heads. A TP8 decode worker therefore replicates
//! every head across 2 ranks, and the 4P1D topology hits this on every request.

use trtllm_core::{Error, Result};

/// The parallel layout on each side of a transfer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reshard {
    pub num_kv_heads: u32,
    pub src_tp: u32,
    pub dst_tp: u32,
}

impl Reshard {
    pub fn identity(num_kv_heads: u32, tp: u32) -> Self {
        Self {
            num_kv_heads,
            src_tp: tp,
            dst_tp: tp,
        }
    }

    pub fn is_identity(&self) -> bool {
        self.src_tp == self.dst_tp
    }

    /// Heads owned by one rank, or 0 when the head is replicated instead.
    pub fn heads_per_rank(&self, tp: u32) -> u32 {
        self.num_kv_heads.checked_div(tp).unwrap_or(0)
    }

    /// How many ranks each head is replicated across on the destination side.
    pub fn dst_replication(&self) -> u32 {
        self.dst_tp
            .checked_div(self.num_kv_heads)
            .unwrap_or(1)
            .max(1)
    }

    /// Build the full rank-to-rank plan.
    pub fn plan(&self) -> Result<ReshardPlan> {
        if self.num_kv_heads == 0 || self.src_tp == 0 || self.dst_tp == 0 {
            return Err(Error::Transfer(
                "reshard needs non-zero heads and TP degrees".into(),
            ));
        }
        if self.src_tp > self.num_kv_heads && self.src_tp % self.num_kv_heads != 0 {
            return Err(Error::Transfer(format!(
                "src TP {} does not divide evenly into {} kv heads",
                self.src_tp, self.num_kv_heads
            )));
        }
        if self.dst_tp > self.num_kv_heads && self.dst_tp % self.num_kv_heads != 0 {
            return Err(Error::Transfer(format!(
                "dst TP {} does not divide evenly into {} kv heads",
                self.dst_tp, self.num_kv_heads
            )));
        }

        let mut edges = Vec::new();
        for head in 0..self.num_kv_heads {
            let src_rank = if self.src_tp <= self.num_kv_heads {
                head / (self.num_kv_heads / self.src_tp)
            } else {
                head * (self.src_tp / self.num_kv_heads)
            };
            if self.dst_tp <= self.num_kv_heads {
                let dst_rank = head / (self.num_kv_heads / self.dst_tp);
                edges.push(ReshardEdge {
                    head,
                    src_rank,
                    dst_rank,
                });
            } else {
                let rep = self.dst_tp / self.num_kv_heads;
                for r in 0..rep {
                    edges.push(ReshardEdge {
                        head,
                        src_rank,
                        dst_rank: head * rep + r,
                    });
                }
            }
        }
        Ok(ReshardPlan {
            reshard: *self,
            edges,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReshardEdge {
    pub head: u32,
    pub src_rank: u32,
    pub dst_rank: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReshardPlan {
    pub reshard: Reshard,
    pub edges: Vec<ReshardEdge>,
}

impl ReshardPlan {
    /// Bytes on the wire relative to a same-TP transfer. Replication makes the
    /// destination want the same head more than once; whether that costs extra
    /// bandwidth depends on whether the fabric can multicast, so this is the
    /// pessimistic figure.
    pub fn amplification(&self) -> f64 {
        self.edges.len() as f64 / f64::from(self.reshard.num_kv_heads.max(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 4P1D case: Qwen3's 4 KV heads, TP2 prefill to TP8 decode.
    #[test]
    fn tp2_to_tp8_replicates_each_head_across_two_ranks() {
        let plan = Reshard {
            num_kv_heads: 4,
            src_tp: 2,
            dst_tp: 8,
        }
        .plan()
        .expect("plan");
        assert_eq!(plan.edges.len(), 8);
        assert_eq!(plan.reshard.dst_replication(), 2);
        assert!((plan.amplification() - 2.0).abs() < 1e-9);

        // Head 3 lives on src rank 1 (heads 2 and 3) and lands on dst ranks 6, 7.
        let head3: Vec<_> = plan.edges.iter().filter(|e| e.head == 3).collect();
        assert_eq!(head3.len(), 2);
        assert!(head3.iter().all(|e| e.src_rank == 1));
        let mut dsts: Vec<u32> = head3.iter().map(|e| e.dst_rank).collect();
        dsts.sort_unstable();
        assert_eq!(dsts, vec![6, 7], "integer division, not modulo");
    }

    #[test]
    fn same_tp_is_one_edge_per_head() {
        let plan = Reshard::identity(4, 4).plan().expect("plan");
        assert_eq!(plan.edges.len(), 4);
        for e in &plan.edges {
            assert_eq!(e.src_rank, e.head);
            assert_eq!(e.dst_rank, e.head);
        }
        assert!((plan.amplification() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn every_destination_rank_is_covered_exactly_once() {
        for dst_tp in [1, 2, 4, 8, 16] {
            let plan = Reshard {
                num_kv_heads: 4,
                src_tp: 2,
                dst_tp,
            }
            .plan()
            .expect("plan");
            let mut ranks: Vec<u32> = plan.edges.iter().map(|e| e.dst_rank).collect();
            ranks.sort_unstable();
            ranks.dedup();
            assert_eq!(
                ranks.len(),
                dst_tp as usize,
                "every dst rank must receive something"
            );
            assert_eq!(
                *ranks.last().expect("edges") + 1,
                dst_tp,
                "ranks must be contiguous from 0"
            );
        }
    }

    #[test]
    fn a_tp_that_does_not_divide_the_heads_is_rejected() {
        assert!(Reshard {
            num_kv_heads: 4,
            src_tp: 2,
            dst_tp: 6
        }
        .plan()
        .is_err());
    }
}
