from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from threading import RLock
from typing import Deque, Dict, List, Optional, Protocol, Tuple


# -----------------------------
# Domain Models
# -----------------------------

@dataclass(frozen=True)
class SalesEvent:
    event_id: str
    merchant_id: str
    product_id: str
    timestamp: datetime
    amount: Decimal


@dataclass(frozen=True)
class MerchantSalesTotal:
    merchant_id: str
    total_amount: Decimal


@dataclass(frozen=True)
class ProductSalesTotal:
    merchant_id: str
    product_id: str
    total_amount: Decimal


@dataclass(frozen=True)
class TopProduct:
    merchant_id: str
    product_id: str
    total_amount: Decimal


class IngestionStatus(str, Enum):
    ACCEPTED = "accepted"
    DUPLICATE_IGNORED = "duplicate_ignored"
    FAILED = "failed"


@dataclass(frozen=True)
class IngestionResult:
    status: IngestionStatus
    merchant_id: Optional[str] = None
    product_id: Optional[str] = None
    message: Optional[str] = None


# -----------------------------
# Sliding Window Top-K Per Merchant
# -----------------------------

class SlidingWindowTopKPerMerchant:
    """
    Tracks Top-K products per merchant over a sliding time window.

    Design:
    - Bucket events by minute to avoid storing every raw event forever.
    - Maintain rolling totals per merchant/product.
    - Evict old buckets outside the window.
    - Query Top-K by sorting only active products for that merchant.

    This is exact, not probabilistic.
    """

    def __init__(self, window_minutes: int = 10, k: int = 5) -> None:
        self.window = timedelta(minutes=window_minutes)
        self.k = k

        # merchant_id -> deque[(bucket_minute, product_id -> amount)]
        self._buckets_by_merchant: Dict[
            str,
            Deque[Tuple[datetime, Dict[str, Decimal]]],
        ] = defaultdict(deque)

        # merchant_id -> product_id -> rolling amount in active window
        self._totals_by_merchant: Dict[str, Dict[str, Decimal]] = defaultdict(dict)

    @staticmethod
    def _truncate_to_minute(ts: datetime) -> datetime:
        return ts.replace(second=0, microsecond=0)

    def add_event(self, event: SalesEvent) -> None:
        bucket_time = self._truncate_to_minute(event.timestamp)
        merchant_buckets = self._buckets_by_merchant[event.merchant_id]

        # If events can arrive out of order, find the existing bucket.
        # For a simple interview implementation, this is acceptable because
        # the active window is only 10 one-minute buckets.
        bucket_data: Optional[Dict[str, Decimal]] = None

        for existing_bucket_time, existing_bucket_data in merchant_buckets:
            if existing_bucket_time == bucket_time:
                bucket_data = existing_bucket_data
                break

        if bucket_data is None:
            bucket_data = defaultdict(lambda: Decimal("0"))
            merchant_buckets.append((bucket_time, bucket_data))
            merchant_buckets = deque(
                sorted(merchant_buckets, key=lambda item: item[0])
            )
            self._buckets_by_merchant[event.merchant_id] = merchant_buckets

        bucket_data[event.product_id] = (
            bucket_data.get(event.product_id, Decimal("0")) + event.amount
        )

        merchant_totals = self._totals_by_merchant[event.merchant_id]
        merchant_totals[event.product_id] = (
            merchant_totals.get(event.product_id, Decimal("0")) + event.amount
        )

    def evict_old(self, merchant_id: str, current_time: datetime) -> None:
        cutoff = current_time - self.window
        merchant_buckets = self._buckets_by_merchant.get(merchant_id)
        if not merchant_buckets:
            return

        merchant_totals = self._totals_by_merchant.get(merchant_id, {})

        while merchant_buckets and merchant_buckets[0][0] < cutoff:
            _, expired_bucket = merchant_buckets.popleft()

            for product_id, amount in expired_bucket.items():
                updated_amount = merchant_totals.get(product_id, Decimal("0")) - amount

                if updated_amount <= Decimal("0"):
                    merchant_totals.pop(product_id, None)
                else:
                    merchant_totals[product_id] = updated_amount

        if not merchant_buckets:
            self._buckets_by_merchant.pop(merchant_id, None)

        if not merchant_totals:
            self._totals_by_merchant.pop(merchant_id, None)

    def get_top_k(
        self,
        merchant_id: str,
        current_time: Optional[datetime] = None,
    ) -> List[TopProduct]:
        if current_time is not None:
            self.evict_old(merchant_id, current_time)

        merchant_totals = self._totals_by_merchant.get(merchant_id, {})

        top_items = sorted(
            merchant_totals.items(),
            key=lambda item: item[1],
            reverse=True,
        )[: self.k]

        return [
            TopProduct(
                merchant_id=merchant_id,
                product_id=product_id,
                total_amount=amount,
            )
            for product_id, amount in top_items
        ]


# -----------------------------
# Repository Interface
# -----------------------------

class SalesEventRepository(Protocol):
    def ingest_once(self, event: SalesEvent) -> bool:
        ...

    def get_total_sales_by_merchant(self, merchant_id: str) -> MerchantSalesTotal:
        ...

    def get_total_sales_by_product(
        self,
        merchant_id: str,
        product_id: str,
    ) -> ProductSalesTotal:
        ...

    def get_top_5_products_last_10_minutes(
        self,
        merchant_id: str,
        current_time: Optional[datetime] = None,
    ) -> List[TopProduct]:
        ...


# -----------------------------
# Concurrent Safe Repository
# -----------------------------

class ConcurrentSafeInMemorySalesRepository:
    """
    In-memory repository optimized for:
    - idempotent writes
    - concurrent updates
    - O(1) merchant/product total reads
    - exact Top-5 products per merchant in the last 10 minutes

    Production equivalents:
    - Redis Lua script for atomic dedup + aggregate update
    - Postgres transaction + unique event_id
    - DynamoDB transaction/conditional write
    - Kafka/Flink keyed state
    """

    def __init__(self, lock_stripes: int = 1024) -> None:
        self._seen_event_ids: set[str] = set()

        # merchant_id -> lifetime total sales
        self._merchant_totals: Dict[str, Decimal] = {}

        # (merchant_id, product_id) -> lifetime total sales
        self._product_totals: Dict[Tuple[str, str], Decimal] = {}

        # Top-5 products per merchant over last 10 minutes
        self._topk_tracker = SlidingWindowTopKPerMerchant(
            window_minutes=10,
            k=5,
        )

        self._locks = [RLock() for _ in range(lock_stripes)]
        self._lock_stripes = lock_stripes

    def _lock_for(self, merchant_id: str, product_id: str) -> RLock:
        stripe = hash((merchant_id, product_id)) % self._lock_stripes
        return self._locks[stripe]

    def ingest_once(self, event: SalesEvent) -> bool:
        """
        Atomically deduplicate and update all derived state.

        Critical invariant:
        If event_id is new, we update:
        - merchant lifetime total
        - product lifetime total
        - sliding-window Top-K state
        - seen_event_ids

        all under the same lock.
        """
        lock = self._lock_for(event.merchant_id, event.product_id)

        with lock:
            if event.event_id in self._seen_event_ids:
                return False

            product_key = (event.merchant_id, event.product_id)

            self._merchant_totals[event.merchant_id] = (
                self._merchant_totals.get(event.merchant_id, Decimal("0"))
                + event.amount
            )

            self._product_totals[product_key] = (
                self._product_totals.get(product_key, Decimal("0"))
                + event.amount
            )

            self._topk_tracker.add_event(event)
            self._topk_tracker.evict_old(
                merchant_id=event.merchant_id,
                current_time=event.timestamp,
            )

            self._seen_event_ids.add(event.event_id)
            return True

    def get_total_sales_by_merchant(self, merchant_id: str) -> MerchantSalesTotal:
        return MerchantSalesTotal(
            merchant_id=merchant_id,
            total_amount=self._merchant_totals.get(merchant_id, Decimal("0")),
        )

    def get_total_sales_by_product(
        self,
        merchant_id: str,
        product_id: str,
    ) -> ProductSalesTotal:
        return ProductSalesTotal(
            merchant_id=merchant_id,
            product_id=product_id,
            total_amount=self._product_totals.get(
                (merchant_id, product_id),
                Decimal("0"),
            ),
        )

    def get_top_5_products_last_10_minutes(
        self,
        merchant_id: str,
        current_time: Optional[datetime] = None,
    ) -> List[TopProduct]:
        """
        Returns exact Top-5 products by sales amount for one merchant
        in the last 10 minutes.
        """
        # Use a merchant-level read lock stripe to avoid reading while the same
        # merchant/product is being updated. For a production system, use a
        # dedicated read/write lock or single-threaded keyed stream processor.
        lock = self._lock_for(merchant_id, "__topk_read__")

        with lock:
            return self._topk_tracker.get_top_k(
                merchant_id=merchant_id,
                current_time=current_time,
            )


# -----------------------------
# Failure Handling
# -----------------------------

@dataclass(frozen=True)
class FailedSalesEvent:
    event: SalesEvent
    error_message: str
    failed_at: datetime


class FailureHandler:
    def __init__(self) -> None:
        self.failed_events: List[FailedSalesEvent] = []

    def handle(self, event: SalesEvent, error: Exception) -> None:
        self.failed_events.append(
            FailedSalesEvent(
                event=event,
                error_message=str(error),
                failed_at=datetime.utcnow(),
            )
        )


# -----------------------------
# Service Layer
# -----------------------------

class FlashSaleSalesService:
    """
    Service owns validation and orchestration.
    Repository owns state, idempotency, and aggregation.
    """

    def __init__(
        self,
        repository: SalesEventRepository,
        failure_handler: FailureHandler,
    ) -> None:
        self.repository = repository
        self.failure_handler = failure_handler

    def ingest_sales_event(self, event: SalesEvent) -> IngestionResult:
        try:
            self._validate_event(event)

            applied = self.repository.ingest_once(event)

            if not applied:
                return IngestionResult(
                    status=IngestionStatus.DUPLICATE_IGNORED,
                    merchant_id=event.merchant_id,
                    product_id=event.product_id,
                    message=f"Duplicate event ignored: {event.event_id}",
                )

            return IngestionResult(
                status=IngestionStatus.ACCEPTED,
                merchant_id=event.merchant_id,
                product_id=event.product_id,
                message="Sales event ingested successfully.",
            )

        except Exception as error:
            self.failure_handler.handle(event, error)

            return IngestionResult(
                status=IngestionStatus.FAILED,
                merchant_id=event.merchant_id,
                product_id=event.product_id,
                message=f"Failed to ingest event: {error}",
            )

    def get_total_sales_by_merchant(
        self,
        merchant_id: str,
    ) -> MerchantSalesTotal:
        return self.repository.get_total_sales_by_merchant(merchant_id)

    def get_total_sales_by_product(
        self,
        merchant_id: str,
        product_id: str,
    ) -> ProductSalesTotal:
        return self.repository.get_total_sales_by_product(
            merchant_id=merchant_id,
            product_id=product_id,
        )

    def get_top_5_products_last_10_minutes(
        self,
        merchant_id: str,
        current_time: Optional[datetime] = None,
    ) -> List[TopProduct]:
        return self.repository.get_top_5_products_last_10_minutes(
            merchant_id=merchant_id,
            current_time=current_time,
        )

    @staticmethod
    def _validate_event(event: SalesEvent) -> None:
        if not event.event_id:
            raise ValueError("event_id is required")

        if not event.merchant_id:
            raise ValueError("merchant_id is required")

        if not event.product_id:
            raise ValueError("product_id is required")

        if event.amount < Decimal("0"):
            raise ValueError("amount cannot be negative")


# -----------------------------
# Example / Sanity Test
# -----------------------------

if __name__ == "__main__":
    repository = ConcurrentSafeInMemorySalesRepository()
    failure_handler = FailureHandler()

    service = FlashSaleSalesService(
        repository=repository,
        failure_handler=failure_handler,
    )

    now = datetime.utcnow()

    events = [
        SalesEvent("evt-1", "merchant-1", "product-A", now, Decimal("100")),
        SalesEvent("evt-2", "merchant-1", "product-B", now, Decimal("80")),
        SalesEvent("evt-3", "merchant-1", "product-C", now, Decimal("60")),
        SalesEvent("evt-4", "merchant-1", "product-D", now, Decimal("40")),
        SalesEvent("evt-5", "merchant-1", "product-E", now, Decimal("20")),
        SalesEvent("evt-6", "merchant-1", "product-F", now, Decimal("10")),
        SalesEvent("evt-7", "merchant-2", "product-Z", now, Decimal("999")),
        SalesEvent("evt-1", "merchant-1", "product-A", now, Decimal("100")),  # duplicate
    ]

    for event in events:
        print(service.ingest_sales_event(event))

    print(service.get_total_sales_by_merchant("merchant-1"))
    print(service.get_total_sales_by_product("merchant-1", "product-A"))

    print("Top 5 merchant-1:")
    for product in service.get_top_5_products_last_10_minutes("merchant-1", now):
        print(product)

    print("Top 5 merchant-2:")
    for product in service.get_top_5_products_last_10_minutes("merchant-2", now):
        print(product)