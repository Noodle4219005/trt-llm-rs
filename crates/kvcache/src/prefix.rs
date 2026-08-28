use std::collections::HashMap;

use trtllm_core::TokenId;

use crate::pool::{BlockId, BlockPool};

/// Chained hash of every token up to and including one block boundary.
///
/// `h_i = fnv1a(h_{i-1} || tokens_of_block_i)`, so two sequences share a hash
/// only when they share the *entire* prefix, not merely one block. Hashing
/// blocks independently is a correctness bug that shows up as garbage output
/// under load, which is why the chain is not optional.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct BlockHash(pub u64);

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a_u64(mut h: u64, v: u64) -> u64 {
    for byte in v.to_le_bytes() {
        h ^= u64::from(byte);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Hash chain over whole blocks of `tokens`. Only complete blocks are hashed;
/// a partially filled tail block is never cacheable because its contents are
/// still changing.
pub fn block_hashes(tokens: &[TokenId], block_size: u32) -> Vec<BlockHash> {
    let bs = block_size.max(1) as usize;
    let mut out = Vec::with_capacity(tokens.len() / bs);
    let mut h = FNV_OFFSET;
    for chunk in tokens.chunks_exact(bs) {
        for &t in chunk {
            h = fnv1a_u64(h, u64::from(t));
        }
        out.push(BlockHash(h));
    }
    out
}

/// What a lookup found.
#[derive(Clone, Debug, Default)]
pub struct PrefixMatch {
    /// Tokens covered by cached blocks - always a multiple of the block size.
    pub num_tokens: usize,
    pub blocks: Vec<BlockId>,
}

impl PrefixMatch {
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

/// Per-worker exact prefix reuse index.
#[derive(Debug)]
pub struct PrefixCache {
    block_size: u32,
    by_hash: HashMap<BlockHash, BlockId>,
    by_block: HashMap<BlockId, BlockHash>,
    hits: u64,
    misses: u64,
    enabled: bool,
}

impl PrefixCache {
    pub fn new(block_size: u32, enabled: bool) -> Self {
        Self {
            block_size,
            by_hash: HashMap::new(),
            by_block: HashMap::new(),
            hits: 0,
            misses: 0,
            enabled,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn len(&self) -> usize {
        self.by_hash.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_hash.is_empty()
    }

    /// Cached-token hit rate. Report it with every run: a result produced at a
    /// non-zero hit rate on a cache-busted workload is measuring the cache,
    /// not the model, and has burned this project once already.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Longest cached prefix of `tokens`, taking a reference on every block it
    /// returns so the pool cannot reclaim them underneath the caller.
    pub fn lookup(&mut self, tokens: &[TokenId], pool: &mut BlockPool) -> PrefixMatch {
        if !self.enabled {
            self.misses += (tokens.len() / self.block_size.max(1) as usize) as u64;
            return PrefixMatch::default();
        }
        let mut m = PrefixMatch::default();
        for h in block_hashes(tokens, self.block_size) {
            match self.by_hash.get(&h) {
                Some(&block) if pool.reacquire(block) => {
                    m.blocks.push(block);
                    m.num_tokens += self.block_size as usize;
                    self.hits += 1;
                }
                Some(&block) => {
                    // The pool already gave this page away; the index entry is
                    // stale, so drop it rather than hand back a wrong block.
                    self.by_hash.remove(&h);
                    self.by_block.remove(&block);
                    self.misses += 1;
                    break;
                }
                None => {
                    self.misses += 1;
                    break;
                }
            }
        }
        m
    }

    /// Publish freshly computed blocks. `tokens` must be the full prefix the
    /// blocks cover, in order, starting at position zero.
    pub fn insert(&mut self, tokens: &[TokenId], blocks: &[BlockId]) {
        if !self.enabled {
            return;
        }
        for (h, &b) in block_hashes(tokens, self.block_size)
            .into_iter()
            .zip(blocks)
        {
            if let Some(prev) = self.by_hash.insert(h, b) {
                if prev != b {
                    self.by_block.remove(&prev);
                }
            }
            self.by_block.insert(b, h);
        }
    }

    /// Drop index entries for blocks the pool has taken back.
    pub fn forget(&mut self, blocks: &[BlockId]) {
        for b in blocks {
            if let Some(h) = self.by_block.remove(b) {
                self.by_hash.remove(&h);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(n: usize, salt: u32) -> Vec<TokenId> {
        (0..n as u32).map(|i| i + salt).collect()
    }

    #[test]
    fn hashes_chain_so_a_shared_block_is_not_a_shared_prefix() {
        let a = block_hashes(&toks(8, 0), 4);
        let mut b_tokens = toks(4, 100);
        b_tokens.extend(toks(4, 4)); // same second block, different first
        let b = block_hashes(&b_tokens, 4);
        assert_eq!(a.len(), 2);
        assert_ne!(a[0], b[0]);
        assert_ne!(
            a[1], b[1],
            "identical block content must still hash differently"
        );
    }

    #[test]
    fn partial_tail_block_is_not_cacheable() {
        assert_eq!(block_hashes(&toks(7, 0), 4).len(), 1);
    }

    #[test]
    fn round_trip_hit_then_forget() {
        let mut pool = BlockPool::new(16, 4, 0.0);
        let mut cache = PrefixCache::new(4, true);
        let tokens = toks(12, 0);
        let blocks = pool.alloc(3).expect("alloc");
        cache.insert(&tokens, &blocks);
        pool.release_to_cache(&blocks);

        let m = cache.lookup(&tokens, &mut pool);
        assert_eq!(m.num_tokens, 12);
        assert_eq!(m.blocks, blocks);

        cache.forget(&blocks);
        pool.release(&blocks);
        let m2 = cache.lookup(&tokens, &mut pool);
        assert!(m2.is_empty());
    }

    #[test]
    fn a_disabled_cache_never_reports_a_hit() {
        let mut pool = BlockPool::new(16, 4, 0.0);
        let mut cache = PrefixCache::new(4, false);
        let tokens = toks(12, 0);
        let blocks = pool.alloc(3).expect("alloc");
        cache.insert(&tokens, &blocks);
        assert!(cache.lookup(&tokens, &mut pool).is_empty());
        assert_eq!(cache.hit_rate(), 0.0);
    }
}
