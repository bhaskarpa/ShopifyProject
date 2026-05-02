import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from decimal import Decimal

from RealtimeFlashSaleMonitor import IngestionStatus, SalesEvent, FlashSaleSalesService, FailureHandler, \
    ConcurrentSafeInMemorySalesRepository


# Replace this import with your actual module/file name
# from flash_sale_sales import (
#     ConcurrentSafeInMemorySalesRepository,
#     FailureHandler,
#     FlashSaleSalesService,
#     IngestionStatus,
#     SalesEvent,
# )


class TestFlashSaleSalesService(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = ConcurrentSafeInMemorySalesRepository()
        self.failure_handler = FailureHandler()
        self.service = FlashSaleSalesService(
            repository=self.repository,
            failure_handler=self.failure_handler,
        )
        self.now = datetime(2026, 1, 1, 12, 0, 0)

    def make_event(
        self,
        event_id: str,
        merchant_id: str = "merchant-1",
        product_id: str = "product-1",
        amount: str = "10.00",
        timestamp: datetime | None = None,
    ) -> SalesEvent:
        return SalesEvent(
            event_id=event_id,
            merchant_id=merchant_id,
            product_id=product_id,
            timestamp=timestamp or self.now,
            amount=Decimal(amount),
        )

    # -----------------------------
    # Basic Ingestion Tests
    # -----------------------------

    def test_single_event_is_accepted(self) -> None:
        result = self.service.ingest_sales_event(self.make_event("evt-1"))

        self.assertEqual(result.status, IngestionStatus.ACCEPTED)

    def test_multiple_events_are_accepted(self) -> None:
        result_1 = self.service.ingest_sales_event(self.make_event("evt-1"))
        result_2 = self.service.ingest_sales_event(self.make_event("evt-2"))

        self.assertEqual(result_1.status, IngestionStatus.ACCEPTED)
        self.assertEqual(result_2.status, IngestionStatus.ACCEPTED)

    def test_merchant_total_is_updated(self) -> None:
        self.service.ingest_sales_event(self.make_event("evt-1", amount="10.00"))
        self.service.ingest_sales_event(self.make_event("evt-2", amount="15.50"))

        total = self.service.get_total_sales_by_merchant("merchant-1")

        self.assertEqual(total.total_amount, Decimal("25.50"))

    def test_product_total_is_updated(self) -> None:
        self.service.ingest_sales_event(
            self.make_event("evt-1", product_id="product-1", amount="10.00")
        )
        self.service.ingest_sales_event(
            self.make_event("evt-2", product_id="product-1", amount="7.25")
        )

        total = self.service.get_total_sales_by_product("merchant-1", "product-1")

        self.assertEqual(total.total_amount, Decimal("17.25"))

    # -----------------------------
    # Idempotency Tests
    # -----------------------------

    def test_duplicate_events_are_ignored(self) -> None:
        event = self.make_event("evt-1", amount="10.00")

        first = self.service.ingest_sales_event(event)
        second = self.service.ingest_sales_event(event)

        self.assertEqual(first.status, IngestionStatus.ACCEPTED)
        self.assertEqual(second.status, IngestionStatus.DUPLICATE_IGNORED)

    def test_duplicate_event_does_not_update_merchant_total(self) -> None:
        event = self.make_event("evt-1", amount="10.00")

        self.service.ingest_sales_event(event)
        self.service.ingest_sales_event(event)

        total = self.service.get_total_sales_by_merchant("merchant-1")

        self.assertEqual(total.total_amount, Decimal("10.00"))

    def test_duplicate_event_does_not_update_product_total(self) -> None:
        event = self.make_event("evt-1", product_id="product-1", amount="10.00")

        self.service.ingest_sales_event(event)
        self.service.ingest_sales_event(event)

        total = self.service.get_total_sales_by_product("merchant-1", "product-1")

        self.assertEqual(total.total_amount, Decimal("10.00"))

    # -----------------------------
    # Validation / Failure Tests
    # -----------------------------

    def test_missing_event_id_fails(self) -> None:
        result = self.service.ingest_sales_event(self.make_event(""))

        self.assertEqual(result.status, IngestionStatus.FAILED)
        self.assertEqual(len(self.failure_handler.failed_events), 1)

    def test_missing_merchant_id_fails(self) -> None:
        result = self.service.ingest_sales_event(
            self.make_event("evt-1", merchant_id="")
        )

        self.assertEqual(result.status, IngestionStatus.FAILED)
        self.assertEqual(len(self.failure_handler.failed_events), 1)

    def test_missing_product_id_fails(self) -> None:
        result = self.service.ingest_sales_event(
            self.make_event("evt-1", product_id="")
        )

        self.assertEqual(result.status, IngestionStatus.FAILED)
        self.assertEqual(len(self.failure_handler.failed_events), 1)

    def test_negative_sale_amount_fails(self) -> None:
        result = self.service.ingest_sales_event(
            self.make_event("evt-1", amount="-1.00")
        )

        self.assertEqual(result.status, IngestionStatus.FAILED)
        self.assertEqual(len(self.failure_handler.failed_events), 1)

    def test_failed_event_does_not_update_totals(self) -> None:
        bad_event = self.make_event("evt-1", amount="-5.00")

        self.service.ingest_sales_event(bad_event)

        merchant_total = self.service.get_total_sales_by_merchant("merchant-1")
        product_total = self.service.get_total_sales_by_product(
            "merchant-1", "product-1"
        )

        self.assertEqual(merchant_total.total_amount, Decimal("0"))
        self.assertEqual(product_total.total_amount, Decimal("0"))

    # -----------------------------
    # Top-5 Per Merchant Tests
    # -----------------------------

    def test_returns_top_5_products_by_amount(self) -> None:
        amounts = {
            "A": "100",
            "B": "90",
            "C": "80",
            "D": "70",
            "E": "60",
            "F": "50",
        }

        for product_id, amount in amounts.items():
            self.service.ingest_sales_event(
                self.make_event(
                    event_id=f"evt-{product_id}",
                    product_id=product_id,
                    amount=amount,
                )
            )

        top = self.service.get_top_5_products_last_10_minutes(
            "merchant-1", self.now
        )

        self.assertEqual([item.product_id for item in top], ["A", "B", "C", "D", "E"])

    def test_top_5_sorted_descending(self) -> None:
        events = [
            ("evt-a", "A", "10"),
            ("evt-b", "B", "50"),
            ("evt-c", "C", "30"),
        ]

        for event_id, product_id, amount in events:
            self.service.ingest_sales_event(
                self.make_event(event_id, product_id=product_id, amount=amount)
            )

        top = self.service.get_top_5_products_last_10_minutes(
            "merchant-1", self.now
        )

        self.assertEqual([item.product_id for item in top], ["B", "C", "A"])
        self.assertEqual([item.total_amount for item in top], [
            Decimal("50"), Decimal("30"), Decimal("10")
        ])

    def test_top_5_scoped_by_merchant(self) -> None:
        self.service.ingest_sales_event(
            self.make_event("evt-1", merchant_id="merchant-1", product_id="A", amount="100")
        )
        self.service.ingest_sales_event(
            self.make_event("evt-2", merchant_id="merchant-2", product_id="Z", amount="999")
        )

        top_merchant_1 = self.service.get_top_5_products_last_10_minutes(
            "merchant-1", self.now
        )

        self.assertEqual(len(top_merchant_1), 1)
        self.assertEqual(top_merchant_1[0].product_id, "A")

    def test_less_than_5_products_returns_all_available(self) -> None:
        for idx, amount in enumerate(["10", "20", "30"], start=1):
            self.service.ingest_sales_event(
                self.make_event(
                    event_id=f"evt-{idx}",
                    product_id=f"product-{idx}",
                    amount=amount,
                )
            )

        top = self.service.get_top_5_products_last_10_minutes(
            "merchant-1", self.now
        )

        self.assertEqual(len(top), 3)

    def test_same_product_receives_multiple_events(self) -> None:
        self.service.ingest_sales_event(
            self.make_event("evt-1", product_id="A", amount="10")
        )
        self.service.ingest_sales_event(
            self.make_event("evt-2", product_id="A", amount="25")
        )

        top = self.service.get_top_5_products_last_10_minutes(
            "merchant-1", self.now
        )

        self.assertEqual(len(top), 1)
        self.assertEqual(top[0].product_id, "A")
        self.assertEqual(top[0].total_amount, Decimal("35"))

    # -----------------------------
    # Sliding Window Tests
    # -----------------------------

    def test_events_inside_10_minutes_are_included(self) -> None:
        self.service.ingest_sales_event(
            self.make_event(
                "evt-1",
                product_id="A",
                amount="100",
                timestamp=self.now - timedelta(minutes=5),
            )
        )

        top = self.service.get_top_5_products_last_10_minutes(
            "merchant-1", self.now
        )

        self.assertEqual(len(top), 1)
        self.assertEqual(top[0].product_id, "A")

    def test_events_older_than_10_minutes_are_evicted(self) -> None:
        self.service.ingest_sales_event(
            self.make_event(
                "evt-old",
                product_id="A",
                amount="100",
                timestamp=self.now - timedelta(minutes=11),
            )
        )
        self.service.ingest_sales_event(
            self.make_event(
                "evt-new",
                product_id="B",
                amount="50",
                timestamp=self.now,
            )
        )

        top = self.service.get_top_5_products_last_10_minutes(
            "merchant-1", self.now
        )

        self.assertEqual([item.product_id for item in top], ["B"])

    def test_boundary_event_at_exactly_10_minutes_is_included(self) -> None:
        self.service.ingest_sales_event(
            self.make_event(
                "evt-boundary",
                product_id="A",
                amount="100",
                timestamp=self.now - timedelta(minutes=10),
            )
        )

        top = self.service.get_top_5_products_last_10_minutes(
            "merchant-1", self.now
        )

        self.assertEqual(len(top), 1)
        self.assertEqual(top[0].product_id, "A")

    def test_out_of_order_timestamp_events_are_bucketed_correctly(self) -> None:
        self.service.ingest_sales_event(
            self.make_event(
                "evt-new",
                product_id="A",
                amount="50",
                timestamp=self.now,
            )
        )
        self.service.ingest_sales_event(
            self.make_event(
                "evt-old-but-valid",
                product_id="B",
                amount="75",
                timestamp=self.now - timedelta(minutes=5),
            )
        )

        top = self.service.get_top_5_products_last_10_minutes(
            "merchant-1", self.now
        )

        self.assertEqual([item.product_id for item in top], ["B", "A"])

    # -----------------------------
    # Concurrency Tests
    # -----------------------------

    def test_concurrent_duplicate_events_are_counted_once(self) -> None:
        event = self.make_event("evt-duplicate", product_id="A", amount="10")

        def ingest() -> IngestionStatus:
            return self.service.ingest_sales_event(event).status

        with ThreadPoolExecutor(max_workers=20) as executor:
            statuses = list(executor.map(lambda _: ingest(), range(100)))

        self.assertEqual(statuses.count(IngestionStatus.ACCEPTED), 1)
        self.assertEqual(statuses.count(IngestionStatus.DUPLICATE_IGNORED), 99)

        merchant_total = self.service.get_total_sales_by_merchant("merchant-1")
        product_total = self.service.get_total_sales_by_product("merchant-1", "A")

        self.assertEqual(merchant_total.total_amount, Decimal("10"))
        self.assertEqual(product_total.total_amount, Decimal("10"))

    def test_concurrent_events_across_different_product_ids(self) -> None:
        events = [
            self.make_event(
                event_id=f"evt-{i}",
                product_id=f"product-{i % 10}",
                amount="1.00",
            )
            for i in range(100)
        ]

        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(self.service.ingest_sales_event, events))

        self.assertEqual(
            sum(1 for result in results if result.status == IngestionStatus.ACCEPTED),
            100,
        )

        merchant_total = self.service.get_total_sales_by_merchant("merchant-1")
        self.assertEqual(merchant_total.total_amount, Decimal("100.00"))

        for i in range(10):
            product_total = self.service.get_total_sales_by_product(
                "merchant-1", f"product-{i}"
            )
            self.assertEqual(product_total.total_amount, Decimal("10.00"))


if __name__ == "__main__":
    unittest.main()
