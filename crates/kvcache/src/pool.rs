use std::fmt;

use serde::Serialize;
use trtllm_core::{Error, Result};

/// Index of one KV page inside a worker's pool.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct BlockId(pub u32);

impl fmt::Debug for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "b{}", self.0)
    }
}

/// Blocks currently backing one sequence.
#[derive(Clone, Debug, Default)]
pub struct SequenceBlocks {
    pub blocks: Vec<BlockId>,
    pub num_tokens: usize,
}

impl SequenceBlocks {
    pub fn capacity_tokens(&self, block_size: u32) -> usize {
        self.blocks.len() * block_size as usize
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct PoolStats {
    pub total: usize,
    pub free: usize,
    pub in_use: usize,
    /// Blocks held only by the prefix cache: allocated, unreferenced, and
    /// reclaimable without preempting anything.
    pub cached: usize,
}

impl PoolStats {
    pub fn utilisation(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.in_use as f64 / self.total as f64
        }
    }
}

/// A paged KV allocator.
///
/// Reference counting is explicit rather than implied by ownership because a
/// block can be shared by several sequences through the prefix cache while
/// also being pinned by the cache itself. `refcount == 0` means "reclaimable",
/// not "free": the block may still hold cached content, and it is the prefix
/// cache that decides when to give it back.
#[derive(Debug)]
pub struct BlockPool {
    block_size: u32,
    refcount: Vec<u32>,
    free: Vec<BlockId>,
    /// Reclaimable blocks that still hold cached content, oldest first.
    cached: std::collections::VecDeque<BlockId>,
    watermark: f64,
}

impl BlockPool {
    pub fn new(num_blocks: u32, block_size: u32, watermark: f64) -> Self {
        let n = num_blocks as usize;
        Self {
            block_size,
            refcount: vec![0; n],
            free: (0..num_blocks).rev().map(BlockId).collect(),
            cached: std::collections::VecDeque::new(),
            watermark: watermark.clamp(0.0, 1.0),
        }
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    pub fn total(&self) -> usize {
        self.refcount.len()
    }

    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }

    /// Blocks that could be handed out right now, counting the reclaimable
    /// cached ones.
    pub fn available(&self) -> usize {
        self.free.len() + self.cached.len()
    }

    pub fn stats(&self) -> PoolStats {
        let in_use = self.refcount.iter().filter(|&&c| c > 0).count();
        PoolStats {
            total: self.total(),
            free: self.free.len(),
            in_use,
            cached: self.cached.len(),
        }
    }

    /// Would admitting a sequence that needs `blocks` pages leave the pool
    /// above its watermark? Admission control asks this before the allocation,
    /// so a sequence is never started and then preempted.
    pub fn can_admit(&self, blocks: usize) -> bool {
        let reserve = (self.total() as f64 * self.watermark).ceil() as usize;
        self.available() >= blocks + reserve
    }

    /// Allocate `n` blocks with a reference already taken.
    pub fn alloc(&mut self, n: usize) -> Result<Vec<BlockId>> {
        if self.available() < n {
            return Err(Error::KvExhausted {
                needed: n,
                free: self.available(),
            });
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let id = match self.free.pop() {
                Some(id) => id,
                // Nothing clean left: take the oldest reclaimable cached block.
                // The caller is the prefix cache's owner, which is responsible
                // for having dropped the corresponding index entry.
                None => self.cached.pop_front().expect("available() checked above"),
            };
            self.refcount[id.0 as usize] = 1;
            out.push(id);
        }
        Ok(out)
    }

    pub fn incref(&mut self, blocks: &[BlockId]) {
        for b in blocks {
            self.refcount[b.0 as usize] += 1;
        }
    }

    /// Drop one reference. Blocks that reach zero go to the *free* list.
    pub fn release(&mut self, blocks: &[BlockId]) {
        for b in blocks {
            let rc = &mut self.refcount[b.0 as usize];
            if *rc > 0 {
                *rc -= 1;
            }
            if *rc == 0 {
                self.free.push(*b);
            }
        }
    }

    /// Drop one reference, keeping the content addressable by the prefix cache
    /// until the pool actually needs the page back.
    pub fn release_to_cache(&mut self, blocks: &[BlockId]) {
        for b in blocks {
            let rc = &mut self.refcount[b.0 as usize];
            if *rc > 0 {
                *rc -= 1;
            }
            if *rc == 0 {
                self.cached.push_back(*b);
            }
        }
    }

    /// Take a cached block back out of the reclaim queue because a new
    /// sequence hit it in the prefix cache.
    pub fn reacquire(&mut self, block: BlockId) -> bool {
        if let Some(pos) = self.cached.iter().position(|b| *b == block) {
            self.cached.remove(pos);
            self.refcount[block.0 as usize] += 1;
            true
        } else if self.refcount[block.0 as usize] > 0 {
            self.refcount[block.0 as usize] += 1;
            true
        } else {
            false
        }
    }

    /// Reclaim up to `n` cached blocks, returning the ones evicted so the
    /// prefix cache can drop their index entries.
    pub fn evict_cached(&mut self, n: usize) -> Vec<BlockId> {
        let mut out = Vec::new();
        for _ in 0..n {
            match self.cached.pop_front() {
                Some(b) => {
                    self.free.push(b);
                    out.push(b);
                }
                None => break,
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_and_release_round_trip() {
        let mut p = BlockPool::new(8, 128, 0.0);
        let a = p.alloc(3).expect("alloc");
        assert_eq!(p.free_blocks(), 5);
        assert_eq!(p.stats().in_use, 3);
        p.release(&a);
        assert_eq!(p.free_blocks(), 8);
        assert_eq!(p.stats().in_use, 0);
    }

    #[test]
    fn exhaustion_is_an_error_not_a_panic() {
        let mut p = BlockPool::new(2, 128, 0.0);
        assert!(p.alloc(3).is_err());
        assert_eq!(p.free_blocks(), 2, "a failed alloc must not consume blocks");
    }

    #[test]
    fn watermark_reserves_headroom_for_running_sequences() {
        let p = BlockPool::new(100, 128, 0.05);
        assert!(p.can_admit(95));
        assert!(!p.can_admit(96));
    }

    #[test]
    fn cached_blocks_are_reclaimable_but_not_free() {
        let mut p = BlockPool::new(4, 128, 0.0);
        let a = p.alloc(4).expect("alloc");
        p.release_to_cache(&a);
        assert_eq!(p.free_blocks(), 0);
        assert_eq!(p.available(), 4);
        let b = p.alloc(2).expect("reclaim from cache");
        assert_eq!(b.len(), 2);
        assert_eq!(p.available(), 2);
    }

    #[test]
    fn shared_blocks_survive_one_release() {
        let mut p = BlockPool::new(4, 128, 0.0);
        let a = p.alloc(2).expect("alloc");
        p.incref(&a);
        p.release(&a);
        assert_eq!(p.stats().in_use, 2, "still referenced by the second holder");
        p.release(&a);
        assert_eq!(p.stats().in_use, 0);
    }
}
