"""Tests for the shadow bridge's sampling.

Both cases here are bugs that shipped. The reservoir was written as
`if len(pool) < cap: pool.append(x)` -- a prefix, not a reservoir, which
reported the warmup ramp as if it were the run (job 316849 read p50 93.77 ms
against a ~15 ms steady state). And the method was once inserted against a
`def snapshot` anchor that does not exist in this file, so the replace silently
did nothing and `STATE.reservoir(...)` would have raised AttributeError on the
first scheduler step of a 16-GPU job.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import trtllm_rs_shadow as shadow


class ShadowSamplingTests(unittest.TestCase):
    def test_reservoir_is_callable_on_the_shared_state(self):
        self.assertTrue(
            hasattr(shadow.STATE, "reservoir"),
            "the live path calls STATE.reservoir every step",
        )

    def test_reservoir_is_not_a_prefix(self):
        """Late samples must be able to displace early ones.

        Feed `cap` samples of 1.0 and then four times as many of 2.0. A prefix
        sampler keeps only the 1.0s and reports a median of 1.0; a uniform
        reservoir ends up roughly 80% 2.0, so the median is 2.0.
        """
        random.seed(2026)
        state = shadow.ShadowState()
        cap = 500
        for _ in range(cap):
            state.reservoir(state.advancing_itl, state.advancing_itl_count, 1.0, cap=cap)
        for _ in range(4 * cap):
            state.reservoir(state.advancing_itl, state.advancing_itl_count, 2.0, cap=cap)

        self.assertEqual(len(state.advancing_itl), cap)
        self.assertEqual(state.advancing_itl_count[0], 5 * cap)
        late = sum(1 for v in state.advancing_itl if v == 2.0) / cap
        self.assertTrue(0.70 < late < 0.90, f"expected ~0.8 late, got {late:.2f}")
        self.assertEqual(shadow._pct(state.advancing_itl, 0.50), 2.0)

    def test_the_two_pools_count_separately(self):
        random.seed(2026)
        state = shadow.ShadowState()
        state.reservoir(state.advancing_itl, state.advancing_itl_count, 1.0)
        state.reservoir(state.steer_itl, state.steer_itl_count, 2.0)
        state.reservoir(state.steer_itl, state.steer_itl_count, 3.0)
        self.assertEqual(state.advancing_itl_count[0], 1)
        self.assertEqual(state.steer_itl_count[0], 2)

    def test_snapshot_reports_both_signals_under_distinct_names(self):
        """The old dump called the advance-only mean `true_itl_ms_ewma`, and
        that name is why it was read as the request-level latency for a whole
        session."""
        keys = shadow.ShadowState().as_dict()
        for k in ("steer_itl_p90", "advancing_itl_p90", "steer_itl_seen"):
            self.assertIn(k, keys)
        self.assertNotIn("true_itl_ms_ewma", keys)


if __name__ == "__main__":
    unittest.main()


class DepartureLedgerTests(unittest.TestCase):
    """A request leaving the candidate list is two different events.

    Job 316849 measured 73.95 requests in the decode phase against 26.3
    candidates offered per step. The forty-eight missing ones passed through
    the pruning loop every step and were dropped, because a request that
    finished its 200 tokens and one that vanished at token 3 both simply
    stopped appearing.
    """

    def test_a_request_that_finished_is_completed(self):
        state = shadow.ShadowState()
        state.token_progress[7] = (0.0, 200, 200)
        state.retire_departed(set())
        self.assertEqual((state.completed, state.stranded), (1, 0))

    def test_a_request_that_vanished_early_is_stranded(self):
        state = shadow.ShadowState()
        state.token_progress[7] = (0.0, 3, 200)
        state.retire_departed(set())
        self.assertEqual((state.completed, state.stranded), (0, 1))
        self.assertEqual(state.stranded_tokens_short, 197)

    def test_a_request_still_being_offered_is_not_retired(self):
        state = shadow.ShadowState()
        state.token_progress[7] = (0.0, 3, 200)
        state.retire_departed({7})
        self.assertEqual((state.completed, state.stranded), (0, 0))
        self.assertIn(7, state.token_progress)

    def test_an_unknown_budget_is_not_counted_as_stranded(self):
        """max_new can come back 0 from a request object that does not expose
        it; guessing "stranded" there would invent the very number this exists
        to measure."""
        state = shadow.ShadowState()
        state.token_progress[7] = (0.0, 3, 0)
        state.retire_departed(set())
        self.assertEqual((state.completed, state.stranded), (1, 0))
