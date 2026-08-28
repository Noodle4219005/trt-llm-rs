//! Router-side approximate prefix index.
//!
//! The router needs one question answered fast: *which worker already holds
//! the longest prefix of this prompt?* It must answer without talking to the
//! workers, so the index is a local, approximate mirror - it stores token ids
//! and worker ids, never block ids, and it is allowed to be wrong. A stale
//! entry costs one recomputed prefix; a missing entry costs nothing but a
//! worse routing choice.
//!
//! With `--cache-bust` in force this index stays empty by construction, which
//! is the correct behaviour for the scored run and the reason the router must
//! never depend on it for load balance.

use std::collections::HashMap;

use trtllm_core::{Millis, TokenId, WorkerId};

#[derive(Debug)]
struct Node {
    /// Tokens on the edge from the parent into this node.
    edge: Vec<TokenId>,
    children: HashMap<TokenId, usize>,
    /// Workers known to hold the prefix ending at this node.
    workers: Vec<WorkerId>,
    last_access_ms: Millis,
}

impl Node {
    fn new(edge: Vec<TokenId>, now: Millis) -> Self {
        Self {
            edge,
            children: HashMap::new(),
            workers: Vec::new(),
            last_access_ms: now,
        }
    }

    fn touch(&mut self, worker: WorkerId, now: Millis) {
        self.last_access_ms = now;
        if !self.workers.contains(&worker) {
            self.workers.push(worker);
        }
    }
}

#[derive(Debug)]
pub struct RadixIndex {
    nodes: Vec<Node>,
    /// Only prefixes at least this long are worth indexing; below it the
    /// bookkeeping costs more than the prefill it could save.
    min_prefix_tokens: usize,
    max_nodes: usize,
}

const ROOT: usize = 0;

impl RadixIndex {
    pub fn new(min_prefix_tokens: usize, max_nodes: usize) -> Self {
        Self {
            nodes: vec![Node::new(Vec::new(), 0.0)],
            min_prefix_tokens,
            max_nodes: max_nodes.max(1),
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.len() <= 1
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.nodes.push(Node::new(Vec::new(), 0.0));
    }

    /// Record that `worker` has computed the KV for `tokens`.
    pub fn insert(&mut self, worker: WorkerId, tokens: &[TokenId], now: Millis) {
        if tokens.len() < self.min_prefix_tokens || self.nodes.len() >= self.max_nodes {
            return;
        }
        let mut cur = ROOT;
        let mut i = 0usize;
        self.nodes[cur].touch(worker, now);

        while i < tokens.len() {
            let first = tokens[i];
            let Some(&child) = self.nodes[cur].children.get(&first) else {
                let node = {
                    let mut n = Node::new(tokens[i..].to_vec(), now);
                    n.touch(worker, now);
                    n
                };
                let id = self.push(node);
                self.nodes[cur].children.insert(first, id);
                return;
            };

            let common = common_prefix_len(&self.nodes[child].edge, &tokens[i..]);
            if common < self.nodes[child].edge.len() {
                self.split(child, common, now);
            }
            i += common;
            cur = child;
            self.nodes[cur].touch(worker, now);
        }
    }

    /// For every worker in the index, the longest prefix of `tokens` it holds.
    /// Workers with no overlap are simply absent.
    pub fn match_workers(&self, tokens: &[TokenId]) -> HashMap<WorkerId, usize> {
        let mut out: HashMap<WorkerId, usize> = HashMap::new();
        let mut cur = ROOT;
        let mut matched = 0usize;

        loop {
            record(&mut out, &self.nodes[cur].workers, matched);
            if matched >= tokens.len() {
                break;
            }
            let Some(&child) = self.nodes[cur].children.get(&tokens[matched]) else {
                break;
            };
            let common = common_prefix_len(&self.nodes[child].edge, &tokens[matched..]);
            matched += common;
            if common < self.nodes[child].edge.len() {
                // Partial edge match. The child's own prefix runs past ours, but
                // a worker holding a longer string necessarily holds every
                // prefix of it - so it still matches the `matched` tokens we
                // share, and stopping without crediting that undercounts.
                record(&mut out, &self.nodes[child].workers, matched);
                break;
            }
            cur = child;
        }
        out
    }

    /// Longest prefix of `tokens` held by `worker`, in tokens.
    pub fn match_worker(&self, worker: WorkerId, tokens: &[TokenId]) -> usize {
        self.match_workers(tokens)
            .get(&worker)
            .copied()
            .unwrap_or(0)
    }

    /// Forget everything a worker knew, e.g. after it restarts or its cache is
    /// flushed. Nodes are kept; only the membership is dropped.
    pub fn forget_worker(&mut self, worker: WorkerId) {
        for n in &mut self.nodes {
            n.workers.retain(|w| *w != worker);
        }
    }

    /// Drop leaves untouched for longer than `ttl_ms`. Called on a timer; the
    /// index is approximate so an imperfect sweep is fine.
    ///
    /// Cost is `O(nodes)` per evicted leaf, which is why `max_nodes` is a hard
    /// cap rather than advice. With `--cache-bust` in force the index holds
    /// nothing and this is free.
    pub fn prune(&mut self, now: Millis, ttl_ms: f64) -> usize {
        let mut removed = 0;
        loop {
            let victim = self.nodes.iter().enumerate().position(|(i, n)| {
                i != ROOT && n.children.is_empty() && now - n.last_access_ms > ttl_ms
            });
            let Some(v) = victim else { break };
            self.detach(v);
            removed += 1;
        }
        removed
    }

    fn push(&mut self, n: Node) -> usize {
        self.nodes.push(n);
        self.nodes.len() - 1
    }

    /// Split `child`'s edge after `at` tokens, pushing the tail into a new node
    /// that inherits the children and the worker set.
    fn split(&mut self, child: usize, at: usize, now: Millis) {
        let tail_edge = self.nodes[child].edge[at..].to_vec();
        let tail_first = tail_edge[0];
        let mut tail = Node::new(tail_edge, now);
        tail.children = std::mem::take(&mut self.nodes[child].children);
        tail.workers = self.nodes[child].workers.clone();
        let tail_id = self.push(tail);

        let node = &mut self.nodes[child];
        node.edge.truncate(at);
        node.children.insert(tail_first, tail_id);
    }

    /// Remove a leaf by swapping the last node into its slot and fixing the one
    /// parent edge that pointed at the moved node.
    fn detach(&mut self, victim: usize) {
        // Unlink the victim from its parent.
        if let Some((parent, key)) = self.find_parent(victim) {
            self.nodes[parent].children.remove(&key);
        }
        let last = self.nodes.len() - 1;
        if victim != last {
            if let Some((parent, key)) = self.find_parent(last) {
                self.nodes[parent].children.insert(key, victim);
            }
            self.nodes.swap(victim, last);
        }
        self.nodes.pop();
    }

    fn find_parent(&self, node: usize) -> Option<(usize, TokenId)> {
        self.nodes.iter().enumerate().find_map(|(i, n)| {
            n.children
                .iter()
                .find(|(_, &c)| c == node)
                .map(|(&k, _)| (i, k))
        })
    }
}

fn record(out: &mut HashMap<WorkerId, usize>, workers: &[WorkerId], matched: usize) {
    for w in workers {
        let e = out.entry(*w).or_insert(0usize);
        *e = (*e).max(matched);
    }
}

fn common_prefix_len(a: &[TokenId], b: &[TokenId]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(v: &[u32]) -> Vec<TokenId> {
        v.to_vec()
    }

    #[test]
    fn exact_prefix_is_found() {
        let mut ix = RadixIndex::new(1, 1024);
        ix.insert(WorkerId(1), &seq(&[1, 2, 3, 4, 5]), 0.0);
        assert_eq!(ix.match_worker(WorkerId(1), &seq(&[1, 2, 3, 4, 5])), 5);
        assert_eq!(ix.match_worker(WorkerId(1), &seq(&[1, 2, 3])), 3);
        assert_eq!(ix.match_worker(WorkerId(2), &seq(&[1, 2, 3])), 0);
    }

    #[test]
    fn a_branch_splits_the_shared_edge() {
        let mut ix = RadixIndex::new(1, 1024);
        ix.insert(WorkerId(1), &seq(&[1, 2, 3, 4]), 0.0);
        ix.insert(WorkerId(2), &seq(&[1, 2, 9, 9]), 1.0);
        assert_eq!(ix.match_worker(WorkerId(1), &seq(&[1, 2, 3, 4])), 4);
        assert_eq!(ix.match_worker(WorkerId(2), &seq(&[1, 2, 9, 9])), 4);
        // Worker 2 shares the [1,2] stem with worker 1 but not the tail.
        assert_eq!(ix.match_worker(WorkerId(2), &seq(&[1, 2, 3, 4])), 2);
        assert_eq!(ix.match_worker(WorkerId(1), &seq(&[1, 2, 9, 9])), 2);
    }

    #[test]
    fn the_best_worker_is_the_one_with_the_longest_overlap() {
        let mut ix = RadixIndex::new(1, 1024);
        ix.insert(WorkerId(1), &seq(&[1, 2]), 0.0);
        ix.insert(WorkerId(2), &seq(&[1, 2, 3, 4, 5, 6]), 0.0);
        let m = ix.match_workers(&seq(&[1, 2, 3, 4, 5, 6, 7]));
        assert_eq!(m[&WorkerId(1)], 2);
        assert_eq!(m[&WorkerId(2)], 6);
    }

    /// A worker that holds a longer, diverging string still holds every token
    /// the two share. Crediting only up to the last full edge silently loses a
    /// whole block of prefix on every near-miss.
    #[test]
    fn a_partial_edge_match_still_counts() {
        let mut ix = RadixIndex::new(1, 1024);
        ix.insert(WorkerId(1), &seq(&[1, 2, 3, 4]), 0.0);
        assert_eq!(ix.match_worker(WorkerId(1), &seq(&[1, 2, 3, 7])), 3);
    }

    #[test]
    fn short_prompts_are_not_indexed() {
        let mut ix = RadixIndex::new(4, 1024);
        ix.insert(WorkerId(1), &seq(&[1, 2]), 0.0);
        assert!(ix.is_empty());
    }

    #[test]
    fn pruning_removes_cold_leaves_and_keeps_the_tree_walkable() {
        let mut ix = RadixIndex::new(1, 1024);
        ix.insert(WorkerId(1), &seq(&[1, 2, 3]), 0.0);
        ix.insert(WorkerId(1), &seq(&[7, 8, 9]), 0.0);
        ix.insert(WorkerId(1), &seq(&[4, 5, 6]), 10_000.0);
        let removed = ix.prune(10_000.0, 5_000.0);
        assert_eq!(removed, 2);
        assert_eq!(ix.match_worker(WorkerId(1), &seq(&[4, 5, 6])), 3);
        assert_eq!(ix.match_worker(WorkerId(1), &seq(&[1, 2, 3])), 0);
    }

    #[test]
    fn forgetting_a_worker_leaves_the_others_intact() {
        let mut ix = RadixIndex::new(1, 1024);
        ix.insert(WorkerId(1), &seq(&[1, 2, 3]), 0.0);
        ix.insert(WorkerId(2), &seq(&[1, 2, 3]), 0.0);
        ix.forget_worker(WorkerId(1));
        assert_eq!(ix.match_worker(WorkerId(1), &seq(&[1, 2, 3])), 0);
        assert_eq!(ix.match_worker(WorkerId(2), &seq(&[1, 2, 3])), 3);
    }
}
