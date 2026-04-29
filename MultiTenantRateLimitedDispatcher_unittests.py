import unittest
from typing import List, Dict, Any

from MultiTenantRateLimitedDispatcher import MultiTenantRateLimitedDispatcher, PartnerConfig, DispatchEvent


# Assumes these are imported from your implementation module.
# from dispatcher import (
#     MultiTenantRateLimitedDispatcher,
#     PartnerConfig,
#     DispatchEvent,
# )


class FakeClock:
    def __init__(self, start: float = 1_000.0):
        self.current = start

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


class RecordingSender:
    def __init__(self, fail_partners=None):
        self.calls: List[Dict[str, Any]] = []
        self.fail_partners = set(fail_partners or [])

    def __call__(self, partner_id: str, events: List["DispatchEvent"]) -> None:
        if partner_id in self.fail_partners:
            raise RuntimeError(f"simulated failure for {partner_id}")

        self.calls.append({
            "partner_id": partner_id,
            "event_ids": [e.event_id for e in events],
            "merchant_ids": [e.merchant_id for e in events],
            "batch_size": len(events),
        })


class MultiTenantRateLimitedDispatcherTests(unittest.TestCase):

    def setUp(self):
        self.clock = FakeClock()
        self.sender = RecordingSender()

        self.configs = [
            PartnerConfig(
                partner_id="google",
                rate_per_second=1.0,
                burst_capacity=1,
                max_batch_size=3,
                max_partner_queue_size=10,
                max_merchant_queue_size=4,
                max_events_per_merchant_per_round=1,
            ),
            PartnerConfig(
                partner_id="meta",
                rate_per_second=2.0,
                burst_capacity=2,
                max_batch_size=2,
                max_partner_queue_size=6,
                max_merchant_queue_size=3,
                max_events_per_merchant_per_round=1,
            ),
        ]

        self.dispatcher = MultiTenantRateLimitedDispatcher(
            partner_configs=self.configs,
            send_batch_fn=self.sender,
            clock=self.clock,
        )

    def event(
        self,
        event_id: str,
        partner_id: str = "google",
        merchant_id: str = "m1",
    ) -> "DispatchEvent":
        return DispatchEvent(
            event_id=event_id,
            partner_id=partner_id,
            merchant_id=merchant_id,
            event_timestamp=self.clock(),
            payload={"event_id": event_id},
        )

    # =========================================================
    # Happy Path
    # =========================================================

    def test_events_enqueue_successfully_in_per_partner_per_merchant_queues(self):
        self.assertTrue(self.dispatcher.submit_event(self.event("g1", "google", "m1")))
        self.assertTrue(self.dispatcher.submit_event(self.event("g2", "google", "m2")))
        self.assertTrue(self.dispatcher.submit_event(self.event("m1", "meta", "m9")))

        google = self.dispatcher.state.partners["google"]
        meta = self.dispatcher.state.partners["meta"]

        self.assertEqual(google.total_buffered_events, 2)
        self.assertEqual(meta.total_buffered_events, 1)

        self.assertIn("m1", google.merchant_queues)
        self.assertIn("m2", google.merchant_queues)
        self.assertIn("m9", meta.merchant_queues)

        self.assertEqual(len(google.merchant_queues["m1"].events), 1)
        self.assertEqual(len(google.merchant_queues["m2"].events), 1)
        self.assertEqual(len(meta.merchant_queues["m9"].events), 1)

    def test_per_partner_batch_flushed_successfully_to_external_partner(self):
        self.dispatcher.submit_event(self.event("g1", "google", "m1"))
        self.dispatcher.submit_event(self.event("g2", "google", "m2"))

        results = self.dispatcher.flush_once()

        self.assertEqual(results["google"].attempted_count, 2)
        self.assertEqual(results["google"].succeeded_count, 2)
        self.assertEqual(results["google"].failed_count, 0)

        self.assertEqual(len(self.sender.calls), 1)
        self.assertEqual(self.sender.calls[0]["partner_id"], "google")
        self.assertCountEqual(self.sender.calls[0]["event_ids"], ["g1", "g2"])

        metrics = self.dispatcher.get_partner_metrics("google")
        self.assertEqual(metrics["buffered_events"], 0)
        self.assertEqual(metrics["total_sent_events"], 2)

    def test_batch_size_respects_partner_constraints(self):
        # google max_batch_size = 3
        for i in range(5):
            self.assertTrue(
                self.dispatcher.submit_event(
                    self.event(f"g{i}", "google", f"m{i}")
                )
            )

        results = self.dispatcher.flush_once()

        self.assertEqual(results["google"].attempted_count, 3)
        self.assertEqual(results["google"].succeeded_count, 3)
        self.assertEqual(self.sender.calls[0]["batch_size"], 3)

        metrics = self.dispatcher.get_partner_metrics("google")
        self.assertEqual(metrics["buffered_events"], 2)

    # =========================================================
    # Rate Limiting
    # =========================================================

    def test_partner_specific_rate_limiting(self):
        # google has burst 1, meta has burst 2.
        self.dispatcher.submit_event(self.event("g1", "google", "g_m1"))
        self.dispatcher.submit_event(self.event("g2", "google", "g_m2"))

        self.dispatcher.submit_event(self.event("m1", "meta", "m_m1"))
        self.dispatcher.submit_event(self.event("m2", "meta", "m_m2"))
        self.dispatcher.submit_event(self.event("m3", "meta", "m_m3"))

        first = self.dispatcher.flush_once()

        self.assertEqual(first["google"].attempted_count, 2)
        self.assertEqual(first["meta"].attempted_count, 2)

        # Add more work immediately.
        self.dispatcher.submit_event(self.event("g3", "google", "g_m3"))
        self.dispatcher.submit_event(self.event("m4", "meta", "m_m4"))

        second = self.dispatcher.flush_once()

        # google had only burst 1 and no time passed, so blocked.
        self.assertEqual(second["google"].attempted_count, 0)

        # meta had burst 2 and consumed only one token in first flush,
        # so it can still send one more batch immediately.
        self.assertEqual(second["meta"].attempted_count, 2)

    def test_second_batch_blocked_if_no_tokens_available(self):
        self.dispatcher.submit_event(self.event("g1", "google", "m1"))
        self.dispatcher.submit_event(self.event("g2", "google", "m2"))

        first = self.dispatcher.flush_once()

        self.dispatcher.submit_event(self.event("g3", "google", "m3"))
        second = self.dispatcher.flush_once()

        self.assertEqual(first["google"].attempted_count, 2)
        self.assertEqual(second["google"].attempted_count, 0)

        # Advance enough time for google to refill one token.
        self.clock.advance(1.1)
        third = self.dispatcher.flush_once()

        self.assertEqual(third["google"].attempted_count, 1)

    # =========================================================
    # Fairness
    # =========================================================

    def test_large_merchant_does_not_starve_other_merchants_for_partner(self):
        # Whale merchant m1 has several events.
        self.dispatcher.submit_event(self.event("g1", "google", "m1"))
        self.dispatcher.submit_event(self.event("g2", "google", "m1"))
        self.dispatcher.submit_event(self.event("g3", "google", "m1"))
        self.dispatcher.submit_event(self.event("g4", "google", "m1"))

        # Smaller merchants have one each.
        self.dispatcher.submit_event(self.event("g5", "google", "m2"))
        self.dispatcher.submit_event(self.event("g6", "google", "m3"))

        result = self.dispatcher.flush_once()

        self.assertEqual(result["google"].attempted_count, 3)

        sent = self.sender.calls[0]
        self.assertEqual(sent["partner_id"], "google")

        # google max_events_per_merchant_per_round = 1, max_batch_size = 3.
        # First batch should include one event from each active merchant.
        self.assertCountEqual(sent["merchant_ids"], ["m1", "m2", "m3"])

    def test_each_merchant_pushes_fair_slice_in_batch_round(self):
        # Rebuild dispatcher with fair slice = 2 for google.
        configs = [
            PartnerConfig(
                partner_id="google",
                rate_per_second=10.0,
                burst_capacity=1,
                max_batch_size=5,
                max_partner_queue_size=20,
                max_merchant_queue_size=10,
                max_events_per_merchant_per_round=2,
            )
        ]

        sender = RecordingSender()
        dispatcher = MultiTenantRateLimitedDispatcher(
            partner_configs=configs,
            send_batch_fn=sender,
            clock=self.clock,
        )

        for i in range(5):
            dispatcher.submit_event(self.event(f"g_m1_{i}", "google", "m1"))

        for i in range(5):
            dispatcher.submit_event(self.event(f"g_m2_{i}", "google", "m2"))

        result = dispatcher.flush_once()

        self.assertEqual(result["google"].attempted_count, 4)

        sent_merchants = sender.calls[0]["merchant_ids"]

        # With two active merchants and slice=2:
        # batch should take two from m1 and two from m2.
        self.assertEqual(sent_merchants.count("m1"), 2)
        self.assertEqual(sent_merchants.count("m2"), 2)

    # =========================================================
    # Partner Isolation
    # =========================================================

    def test_google_and_meta_operate_independently(self):
        self.dispatcher.submit_event(self.event("g1", "google", "gm1"))
        self.dispatcher.submit_event(self.event("m1", "meta", "mm1"))

        results = self.dispatcher.flush_once()

        self.assertEqual(results["google"].succeeded_count, 1)
        self.assertEqual(results["meta"].succeeded_count, 1)

        partners_sent = [call["partner_id"] for call in self.sender.calls]
        self.assertIn("google", partners_sent)
        self.assertIn("meta", partners_sent)

    def test_problem_with_one_partner_does_not_stop_other_partner_flush(self):
        sender = RecordingSender(fail_partners={"google"})

        dispatcher = MultiTenantRateLimitedDispatcher(
            partner_configs=self.configs,
            send_batch_fn=sender,
            clock=self.clock,
        )

        dispatcher.submit_event(self.event("g1", "google", "gm1"))
        dispatcher.submit_event(self.event("m1", "meta", "mm1"))

        results = dispatcher.flush_once()

        self.assertEqual(results["google"].attempted_count, 1)
        self.assertEqual(results["google"].succeeded_count, 0)
        self.assertEqual(results["google"].failed_count, 1)

        self.assertEqual(results["meta"].attempted_count, 1)
        self.assertEqual(results["meta"].succeeded_count, 1)
        self.assertEqual(results["meta"].failed_count, 0)

        google_metrics = dispatcher.get_partner_metrics("google")
        meta_metrics = dispatcher.get_partner_metrics("meta")

        self.assertEqual(google_metrics["dlq_size"], 1)
        self.assertEqual(meta_metrics["total_sent_events"], 1)

    # =========================================================
    # Backpressure Handling
    # =========================================================

    def test_per_merchant_queue_full_rejects_additional_events_for_that_merchant(self):
        # google max_merchant_queue_size = 4
        self.assertTrue(self.dispatcher.submit_event(self.event("g1", "google", "m1")))
        self.assertTrue(self.dispatcher.submit_event(self.event("g2", "google", "m1")))
        self.assertTrue(self.dispatcher.submit_event(self.event("g3", "google", "m1")))
        self.assertTrue(self.dispatcher.submit_event(self.event("g4", "google", "m1")))

        self.assertFalse(self.dispatcher.submit_event(self.event("g5", "google", "m1")))

        metrics = self.dispatcher.get_partner_metrics("google")
        self.assertEqual(metrics["buffered_events"], 4)
        self.assertEqual(metrics["total_dropped_events"], 1)

    def test_per_partner_queue_full_rejects_events_across_all_merchants(self):
        # google max_partner_queue_size = 10
        accepted = 0
        rejected = 0

        for i in range(12):
            ok = self.dispatcher.submit_event(
                self.event(f"g{i}", "google", f"m{i}")  # unique merchants
            )
            if ok:
                accepted += 1
            else:
                rejected += 1

        self.assertEqual(accepted, 10)
        self.assertEqual(rejected, 2)

        metrics = self.dispatcher.get_partner_metrics("google")
        self.assertEqual(metrics["buffered_events"], 10)
        self.assertEqual(metrics["total_dropped_events"], 2)


if __name__ == "__main__":
    unittest.main()