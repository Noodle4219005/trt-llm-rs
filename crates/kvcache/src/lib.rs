//! KV cache management.
//!
//! Three separate things live here, and keeping them separate matters:
//!
//! * [`pool::BlockPool`] is the per-worker paged allocator. It is the only
//!   thing that owns GPU memory, and every admission decision has to go
//!   through its watermark.
//! * [`prefix::PrefixCache`] is the per-worker exact reuse index, keyed on a
//!   hash chain over whole blocks, with LRU eviction over unreferenced blocks.
//! * [`radix::RadixIndex`] is the router-side *approximate* view of which
//!   worker holds which prefix. It stores token ids, never blocks, and exists
//!   only to score routing candidates.
//!
//! Note that the scored benchmark for this deployment runs with the prefix
//! busted on purpose, so the reuse paths must never be on the critical path of
//! a number we report. They are here because a serving framework needs them,
//! not because they help the score.

pub mod pool;
pub mod prefix;
pub mod radix;

pub use pool::{BlockId, BlockPool, PoolStats, SequenceBlocks};
pub use prefix::{BlockHash, PrefixCache, PrefixMatch};
pub use radix::RadixIndex;

/// Number of blocks needed to hold `tokens` tokens at `block_size`.
#[inline]
pub fn blocks_for(tokens: usize, block_size: u32) -> usize {
    let bs = block_size.max(1) as usize;
    tokens.div_ceil(bs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_count_rounds_up() {
        assert_eq!(blocks_for(0, 128), 0);
        assert_eq!(blocks_for(1, 128), 1);
        assert_eq!(blocks_for(128, 128), 1);
        assert_eq!(blocks_for(129, 128), 2);
        assert_eq!(blocks_for(4000, 128), 32);
    }
}
