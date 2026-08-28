use std::cmp::Ordering;

use trtllm_core::{Millis, RequestId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Event {
    /// A closed-loop client issues its next request.
    Arrival { client: usize },
    /// A prefill worker finished the batch it was running.
    PrefillDone { worker: usize },
    /// A request's KV has landed on its decode worker.
    KvArrived { id: RequestId, worker: usize },
    /// A decode worker completed one forward pass.
    DecodeStep { worker: usize },
}

/// Heap entry ordered by time, then by an insertion sequence number so that two
/// events at the same instant always fire in the same order. Determinism here
/// is not a nicety: without it an A/B between two policies is noise.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Scheduled {
    pub at: Millis,
    pub seq: u64,
    pub event: Event,
}

impl PartialEq for Scheduled {
    fn eq(&self, other: &Self) -> bool {
        self.at == other.at && self.seq == other.seq
    }
}
impl Eq for Scheduled {}

impl Ord for Scheduled {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed: BinaryHeap is a max-heap and we want the earliest event.
        other.at.total_cmp(&self.at).then(other.seq.cmp(&self.seq))
    }
}

impl PartialOrd for Scheduled {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BinaryHeap;

    #[test]
    fn the_heap_pops_in_time_then_insertion_order() {
        let mut h = BinaryHeap::new();
        h.push(Scheduled {
            at: 10.0,
            seq: 1,
            event: Event::PrefillDone { worker: 0 },
        });
        h.push(Scheduled {
            at: 5.0,
            seq: 2,
            event: Event::PrefillDone { worker: 1 },
        });
        h.push(Scheduled {
            at: 5.0,
            seq: 0,
            event: Event::PrefillDone { worker: 2 },
        });
        let order: Vec<u64> = std::iter::from_fn(|| h.pop()).map(|s| s.seq).collect();
        assert_eq!(order, vec![0, 2, 1]);
    }
}
