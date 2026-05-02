import time
import unittest

from DistributedWebhookIngestor import WebhookConsumer, InMemoryUnitOfWork, InMemoryTaskQueue, WebhookEndpoint, \
    WebhookStatus, InMemoryGlobalLockStore, InMemoryDatabase, WebhookEvent


# Assumes imports from your implementation module:
# from webhook_ingestor import (
#     WebhookEvent,
#     WebhookEndpoint,
#     WebhookConsumer,
#     InMemoryDatabase,
#     InMemoryTaskQueue,
#     InMemoryUnitOfWork,
#     InMemoryGlobalLockStore,
#     WebhookStatus,
# )


class WebhookIngestorTests(unittest.TestCase):

    def make_event(
        self,
        provider_event_id="evt_1",
        provider="stripe",
        event_type="payment.succeeded",
        amount=100.0,
        currency="USD",
    ):
        return WebhookEvent(
            provider_event_id=provider_event_id,
            provider=provider,
            event_type=event_type,
            payload={"amount": amount, "currency": currency},
        )

    def setUp(self):
        self.db = InMemoryDatabase()
        self.global_lock = InMemoryGlobalLockStore()
        self.queue = InMemoryTaskQueue(max_attempts=5, base_backoff_seconds=0.01)

        self.endpoint = WebhookEndpoint(self.queue)
        self.consumer = WebhookConsumer(
            queue=self.queue,
            uow_factory=lambda: InMemoryUnitOfWork(self.db),
            global_lock_store=self.global_lock,
            lock_ttl_seconds=60,
            region_id="region-a",
        )

    def drain_queue(self, consumer, attempts=10, sleep_seconds=0.02):
        for _ in range(attempts):
            processed = consumer.process_next()
            if processed:
                continue
            time.sleep(sleep_seconds)

    # =========================================================
    # Basic Ingestion Tests
    # =========================================================

    def test_webhook_event_is_accepted(self):
        event = self.make_event()

        result = self.endpoint.ingest_request(event)

        self.assertTrue(result.accepted)
        self.assertEqual(len(self.db.webhook_events), 0)

    def test_multiple_different_webhook_events_are_accepted(self):
        event1 = self.make_event("evt_1")
        event2 = self.make_event("evt_2")

        self.endpoint.ingest_request(event1)
        self.endpoint.ingest_request(event2)

        self.drain_queue(self.consumer)

        self.assertEqual(len(self.db.webhook_events), 2)
        self.assertEqual(len(self.db.payment_impacts), 2)

    def test_payment_is_processed(self):
        event = self.make_event("evt_payment_1", amount=125.50)

        self.endpoint.ingest_request(event)
        self.drain_queue(self.consumer)

        payment = self.db.payment_impacts["evt_payment_1"]

        self.assertEqual(payment.amount, 125.50)
        self.assertEqual(payment.currency, "USD")
        self.assertEqual(payment.provider, "stripe")

    def test_event_is_stored_to_database(self):
        event = self.make_event("evt_store_1")

        self.endpoint.ingest_request(event)
        self.drain_queue(self.consumer)

        stored = self.db.webhook_events["evt_store_1"]

        self.assertEqual(stored.provider_event_id, "evt_store_1")
        self.assertEqual(stored.provider, "stripe")
        self.assertEqual(stored.status, WebhookStatus.PROCESSED)

    def test_events_from_different_payment_providers_are_stored(self):
        stripe_event = self.make_event(
            provider_event_id="stripe_evt_1",
            provider="stripe",
        )
        paypal_event = self.make_event(
            provider_event_id="paypal_evt_1",
            provider="paypal",
        )

        self.endpoint.ingest_request(stripe_event)
        self.endpoint.ingest_request(paypal_event)

        self.drain_queue(self.consumer)

        self.assertIn("stripe_evt_1", self.db.webhook_events)
        self.assertIn("paypal_evt_1", self.db.webhook_events)

        self.assertEqual(self.db.webhook_events["stripe_evt_1"].provider, "stripe")
        self.assertEqual(self.db.webhook_events["paypal_evt_1"].provider, "paypal")

    # =========================================================
    # Idempotency Tests
    # =========================================================

    def test_duplicate_events_from_same_provider_are_ignored(self):
        event = self.make_event("evt_duplicate")

        self.endpoint.ingest_request(event)
        self.endpoint.ingest_request(event)

        self.drain_queue(self.consumer)

        self.assertEqual(len(self.db.webhook_events), 1)
        self.assertEqual(len(self.db.payment_impacts), 1)

    def test_duplicate_event_should_not_create_second_payment_record(self):
        event = self.make_event("evt_payment_duplicate", amount=200.0)

        self.endpoint.ingest_request(event)
        self.endpoint.ingest_request(event)

        self.drain_queue(self.consumer)

        self.assertEqual(len(self.db.payment_impacts), 1)
        self.assertIn("evt_payment_duplicate", self.db.payment_impacts)

    # =========================================================
    # Out-of-Order Event Tests
    # =========================================================

    def test_handle_out_of_order_events(self):
        """
        For this simplified payment.succeeded-only implementation, events are
        independently idempotent. Out-of-order delivery should not corrupt state.
        """
        event2 = self.make_event("evt_2", amount=200.0)
        event1 = self.make_event("evt_1", amount=100.0)

        self.endpoint.ingest_request(event2)
        self.endpoint.ingest_request(event1)

        self.drain_queue(self.consumer)

        self.assertEqual(len(self.db.webhook_events), 2)
        self.assertEqual(len(self.db.payment_impacts), 2)
        self.assertIn("evt_1", self.db.payment_impacts)
        self.assertIn("evt_2", self.db.payment_impacts)

    # =========================================================
    # Database Failure Tests
    # =========================================================

    def test_database_unavailable_during_write_event_should_not_be_lost(self):
        event = self.make_event("evt_db_down")

        self.endpoint.ingest_request(event)

        self.db.fail_next_operations = 1
        processed = self.consumer.process_next()

        self.assertFalse(processed)
        self.assertEqual(len(self.db.webhook_events), 0)
        self.assertEqual(len(self.queue.dead_letters), 0)

    def test_retry_succeeds_after_database_recovers(self):
        event = self.make_event("evt_retry_success")

        self.endpoint.ingest_request(event)

        self.db.fail_next_operations = 1
        first = self.consumer.process_next()
        self.assertFalse(first)

        time.sleep(0.02)

        second = self.consumer.process_next()
        self.assertTrue(second)

        self.assertIn("evt_retry_success", self.db.webhook_events)
        self.assertIn("evt_retry_success", self.db.payment_impacts)

    def test_event_is_processed_exactly_once_after_retry(self):
        event = self.make_event("evt_exactly_once", amount=333.0)

        self.endpoint.ingest_request(event)

        self.db.fail_next_operations = 1
        self.consumer.process_next()

        time.sleep(0.02)

        self.consumer.process_next()
        self.consumer.process_next()

        self.assertEqual(len(self.db.webhook_events), 1)
        self.assertEqual(len(self.db.payment_impacts), 1)
        self.assertEqual(self.db.payment_impacts["evt_exactly_once"].amount, 333.0)

    # =========================================================
    # Multi-Region Tests
    # =========================================================

    def test_same_event_hits_region_a_and_region_b(self):
        queue_a = InMemoryTaskQueue(max_attempts=5, base_backoff_seconds=0.01)
        queue_b = InMemoryTaskQueue(max_attempts=5, base_backoff_seconds=0.01)

        endpoint_a = WebhookEndpoint(queue_a)
        endpoint_b = WebhookEndpoint(queue_b)

        consumer_a = WebhookConsumer(
            queue=queue_a,
            uow_factory=lambda: InMemoryUnitOfWork(self.db),
            global_lock_store=self.global_lock,
            lock_ttl_seconds=60,
            region_id="region-a",
        )
        consumer_b = WebhookConsumer(
            queue=queue_b,
            uow_factory=lambda: InMemoryUnitOfWork(self.db),
            global_lock_store=self.global_lock,
            lock_ttl_seconds=60,
            region_id="region-b",
        )

        event = self.make_event("evt_multiregion")

        endpoint_a.ingest_request(event)
        endpoint_b.ingest_request(event)

        consumer_a.process_next()
        consumer_b.process_next()

        self.assertEqual(len(self.db.webhook_events), 1)
        self.assertEqual(len(self.db.payment_impacts), 1)

    def test_only_one_region_wins_the_idempotency_claim(self):
        queue_a = InMemoryTaskQueue(max_attempts=5, base_backoff_seconds=0.01)
        queue_b = InMemoryTaskQueue(max_attempts=5, base_backoff_seconds=0.01)

        queue_a.publish(self.make_event("evt_claim"))
        queue_b.publish(self.make_event("evt_claim"))

        consumer_a = WebhookConsumer(
            queue=queue_a,
            uow_factory=lambda: InMemoryUnitOfWork(self.db),
            global_lock_store=self.global_lock,
            lock_ttl_seconds=60,
            region_id="region-a",
        )
        consumer_b = WebhookConsumer(
            queue=queue_b,
            uow_factory=lambda: InMemoryUnitOfWork(self.db),
            global_lock_store=self.global_lock,
            lock_ttl_seconds=60,
            region_id="region-b",
        )

        consumer_a.process_next()
        consumer_b.process_next()

        self.assertEqual(len(self.db.webhook_events), 1)
        self.assertEqual(len(self.db.payment_impacts), 1)

    def test_global_idempotency_conflict_is_handled_correctly(self):
        event = self.make_event("evt_lock_conflict")

        lock_key = f"webhook:{event.provider}:{event.provider_event_id}"
        acquired = self.global_lock.acquire(
            key=lock_key,
            owner="other-region",
            ttl_seconds=60,
        )
        self.assertTrue(acquired)

        self.queue.publish(event)

        processed = self.consumer.process_next()

        self.assertFalse(processed)
        self.assertEqual(len(self.db.webhook_events), 0)
        self.assertEqual(len(self.db.payment_impacts), 0)
        self.assertEqual(len(self.queue.dead_letters), 0)

    def test_region_failure_replays_events_without_double_processing(self):
        """
        Region A fails during DB write, requeues the task.
        Region B later processes the same event.
        Region A retry should not double-process it.
        """
        queue_a = InMemoryTaskQueue(max_attempts=5, base_backoff_seconds=0.01)
        queue_b = InMemoryTaskQueue(max_attempts=5, base_backoff_seconds=0.01)

        event = self.make_event("evt_region_replay", amount=500.0)

        queue_a.publish(event)
        queue_b.publish(event)

        consumer_a = WebhookConsumer(
            queue=queue_a,
            uow_factory=lambda: InMemoryUnitOfWork(self.db),
            global_lock_store=self.global_lock,
            lock_ttl_seconds=60,
            region_id="region-a",
        )
        consumer_b = WebhookConsumer(
            queue=queue_b,
            uow_factory=lambda: InMemoryUnitOfWork(self.db),
            global_lock_store=self.global_lock,
            lock_ttl_seconds=60,
            region_id="region-b",
        )

        self.db.fail_next_operations = 1
        first = consumer_a.process_next()
        self.assertFalse(first)

        second = consumer_b.process_next()
        self.assertTrue(second)

        time.sleep(0.02)

        # Region A retry sees the event already processed and exits safely.
        retry = consumer_a.process_next()
        self.assertTrue(retry)

        self.assertEqual(len(self.db.webhook_events), 1)
        self.assertEqual(len(self.db.payment_impacts), 1)
        self.assertEqual(self.db.payment_impacts["evt_region_replay"].amount, 500.0)


if __name__ == "__main__":
    unittest.main()
