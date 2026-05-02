from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Protocol, List
import heapq
import time
import uuid
import random


# ============================================================
# Models
# ============================================================

class WebhookStatus(str, Enum):
    RECEIVED = "received"
    PROCESSED = "processed"
    FAILED = "failed"


@dataclass(frozen=True)
class WebhookEvent:
    provider_event_id: str
    provider: str
    event_type: str
    payload: Dict[str, Any]
    received_at: float = field(default_factory=time.time)


@dataclass
class StoredWebhookEvent:
    id: str
    provider_event_id: str
    provider: str
    event_type: str
    payload: Dict[str, Any]
    status: WebhookStatus
    received_at: float
    processed_at: Optional[float] = None
    failure_reason: Optional[str] = None


@dataclass
class PaymentImpactRecord:
    id: str
    provider_event_id: str
    provider: str
    amount: float
    currency: str
    created_at: float


@dataclass(frozen=True)
class IngestResult:
    accepted: bool
    message: str


# ============================================================
# Queue Models
# ============================================================

@dataclass(order=True)
class QueuedTask:
    available_at: float
    task_id: str = field(compare=False)
    event: WebhookEvent = field(compare=False)
    attempt: int = field(default=0, compare=False)
    last_error: Optional[str] = field(default=None, compare=False)


class TaskQueue(Protocol):
    def publish(self, event: WebhookEvent) -> None:
        ...

    def poll(self) -> Optional[QueuedTask]:
        ...

    def retry_later(self, task: QueuedTask, error: str) -> None:
        ...

    def dead_letter(self, task: QueuedTask, error: str) -> None:
        ...


class InMemoryTaskQueue:
    """
    Simple priority-queue based task queue.

    In production this would be Kafka, SQS, Pub/Sub, RabbitMQ, etc.
    available_at allows delayed retries.
    """

    def __init__(
        self,
        max_attempts: int = 5,
        base_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 60.0,
    ) -> None:
        self.max_attempts = max_attempts
        self.base_backoff_seconds = base_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds

        self._queue: List[QueuedTask] = []
        self.dead_letters: List[QueuedTask] = []

    def publish(self, event: WebhookEvent) -> None:
        task = QueuedTask(
            available_at=time.time(),
            task_id=str(uuid.uuid4()),
            event=event,
            attempt=0,
        )
        heapq.heappush(self._queue, task)

    def poll(self) -> Optional[QueuedTask]:
        if not self._queue:
            return None

        task = self._queue[0]
        if task.available_at > time.time():
            return None

        return heapq.heappop(self._queue)

    def retry_later(self, task: QueuedTask, error: str) -> None:
        next_attempt = task.attempt + 1

        if next_attempt >= self.max_attempts:
            self.dead_letter(task, error)
            return

        backoff = min(
            self.max_backoff_seconds,
            self.base_backoff_seconds * (2 ** task.attempt),
        )

        # Jitter prevents many failed tasks from retrying at the same time.
        jitter = random.uniform(0, backoff * 0.2)

        retry_task = QueuedTask(
            available_at=time.time() + backoff + jitter,
            task_id=task.task_id,
            event=task.event,
            attempt=next_attempt,
            last_error=error,
        )

        heapq.heappush(self._queue, retry_task)

    def dead_letter(self, task: QueuedTask, error: str) -> None:
        dead_task = QueuedTask(
            available_at=time.time(),
            task_id=task.task_id,
            event=task.event,
            attempt=task.attempt,
            last_error=error,
        )
        self.dead_letters.append(dead_task)


# ============================================================
# Repository Interfaces
# ============================================================

class WebhookEventRepository(Protocol):
    def find_by_provider_event_id(
        self,
        provider_event_id: str,
    ) -> Optional[StoredWebhookEvent]:
        ...

    def insert_event(self, event: WebhookEvent) -> StoredWebhookEvent:
        ...

    def update_status(
        self,
        provider_event_id: str,
        status: WebhookStatus,
        failure_reason: Optional[str] = None,
    ) -> None:
        ...


class PaymentRepository(Protocol):
    def insert_payment_impact(
        self,
        provider_event_id: str,
        provider: str,
        amount: float,
        currency: str,
    ) -> PaymentImpactRecord:
        ...


class UnitOfWork(Protocol):
    webhook_events: WebhookEventRepository
    payments: PaymentRepository

    def __enter__(self) -> "UnitOfWork":
        ...

    def __exit__(self, exc_type, exc, tb) -> None:
        ...


# ============================================================
# In-Memory SQL-like Database
# ============================================================

class TransientDatabaseError(Exception):
    pass


class InMemoryDatabase:
    def __init__(self) -> None:
        self.webhook_events: Dict[str, StoredWebhookEvent] = {}
        self.payment_impacts: Dict[str, PaymentImpactRecord] = {}

        # For tests/demo. Set this > 0 to simulate transient DB failure.
        self.fail_next_operations = 0

    def maybe_fail(self) -> None:
        if self.fail_next_operations > 0:
            self.fail_next_operations -= 1
            raise TransientDatabaseError("temporary database failure")


class InMemoryWebhookEventRepository:
    def __init__(self, db: InMemoryDatabase) -> None:
        self.db = db

    def find_by_provider_event_id(
        self,
        provider_event_id: str,
    ) -> Optional[StoredWebhookEvent]:
        self.db.maybe_fail()
        return self.db.webhook_events.get(provider_event_id)

    def insert_event(self, event: WebhookEvent) -> StoredWebhookEvent:
        self.db.maybe_fail()

        if event.provider_event_id in self.db.webhook_events:
            raise ValueError("duplicate provider_event_id")

        stored = StoredWebhookEvent(
            id=str(uuid.uuid4()),
            provider_event_id=event.provider_event_id,
            provider=event.provider,
            event_type=event.event_type,
            payload=event.payload,
            status=WebhookStatus.RECEIVED,
            received_at=event.received_at,
        )

        self.db.webhook_events[event.provider_event_id] = stored
        return stored

    def update_status(
        self,
        provider_event_id: str,
        status: WebhookStatus,
        failure_reason: Optional[str] = None,
    ) -> None:
        self.db.maybe_fail()

        stored = self.db.webhook_events[provider_event_id]
        stored.status = status
        stored.failure_reason = failure_reason

        if status == WebhookStatus.PROCESSED:
            stored.processed_at = time.time()


class InMemoryPaymentRepository:
    def __init__(self, db: InMemoryDatabase) -> None:
        self.db = db

    def insert_payment_impact(
        self,
        provider_event_id: str,
        provider: str,
        amount: float,
        currency: str,
    ) -> PaymentImpactRecord:
        self.db.maybe_fail()

        # Independent idempotency guard for the payment-impacting side effect.
        if provider_event_id in self.db.payment_impacts:
            raise ValueError("duplicate payment impact")

        record = PaymentImpactRecord(
            id=str(uuid.uuid4()),
            provider_event_id=provider_event_id,
            provider=provider,
            amount=amount,
            currency=currency,
            created_at=time.time(),
        )

        self.db.payment_impacts[provider_event_id] = record
        return record


class InMemoryUnitOfWork:
    def __init__(self, db: InMemoryDatabase) -> None:
        self.db = db
        self.webhook_events = InMemoryWebhookEventRepository(db)
        self.payments = InMemoryPaymentRepository(db)

    def __enter__(self) -> "InMemoryUnitOfWork":
        # Production: begin transaction.
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Production:
        #   if exc: rollback
        #   else: commit
        return None


# ============================================================
# Global Lock / Idempotency Store
# ============================================================

class GlobalLockStore(Protocol):
    def acquire(
        self,
        key: str,
        owner: str,
        ttl_seconds: int,
    ) -> bool:
        ...

    def release(
        self,
        key: str,
        owner: str,
    ) -> None:
        ...


@dataclass
class LockRecord:
    owner: str
    expires_at: float


class InMemoryGlobalLockStore:
    """
    Redis-like SET NX EX behavior.

    Production Redis equivalent:
        SET lock_key owner NX EX ttl_seconds

    Release should be owner-checked to avoid deleting another worker's lock.
    """

    def __init__(self) -> None:
        self.locks: dict[str, LockRecord] = {}

    def acquire(
        self,
        key: str,
        owner: str,
        ttl_seconds: int,
    ) -> bool:
        now = time.time()

        existing = self.locks.get(key)

        if existing and existing.expires_at > now:
            return False

        self.locks[key] = LockRecord(
            owner=owner,
            expires_at=now + ttl_seconds,
        )
        return True

    def release(
        self,
        key: str,
        owner: str,
    ) -> None:
        existing = self.locks.get(key)

        if existing and existing.owner == owner:
            del self.locks[key]


# ============================================================
# Endpoint Layer
# ============================================================

class WebhookEndpoint:
    """
    Thin HTTP-facing layer.

    It does not touch the database.
    It only validates schema and publishes a durable task.
    """

    def __init__(self, queue: TaskQueue) -> None:
        self.queue = queue

    def ingest_request(self, event: WebhookEvent) -> IngestResult:
        self._validate_schema(event)

        self.queue.publish(event)

        return IngestResult(
            accepted=True,
            message="Webhook accepted for async processing",
        )

    def _validate_schema(self, event: WebhookEvent) -> None:
        if not event.provider_event_id:
            raise ValueError("provider_event_id is required")
        if not event.provider:
            raise ValueError("provider is required")
        if not event.event_type:
            raise ValueError("event_type is required")
        if event.payload is None:
            raise ValueError("payload is required")


# ============================================================
# Consumer Service
# ============================================================

class WebhookConsumer:
    """
    Multi-region safe queue consumer.

    Global consistency strategy:
      1. Acquire global lock using provider + provider_event_id
      2. Check durable DB idempotency
      3. Process payment-impacting side effect
      4. Mark event processed
      5. Release lock

    The global lock prevents two regions from processing the same event at the
    same time. The database uniqueness check remains the final safety net.
    """

    def __init__(
        self,
        queue: TaskQueue,
        uow_factory,
        global_lock_store: GlobalLockStore,
        lock_ttl_seconds: int = 60,
        region_id: str = "unknown-region",
    ) -> None:
        self.queue = queue
        self.uow_factory = uow_factory
        self.global_lock_store = global_lock_store
        self.lock_ttl_seconds = lock_ttl_seconds
        self.region_id = region_id

    def process_next(self) -> bool:
        task = self.queue.poll()
        if task is None:
            return False

        lock_key = self._lock_key(task.event)
        lock_owner = self._lock_owner(task)

        acquired = self.global_lock_store.acquire(
            key=lock_key,
            owner=lock_owner,
            ttl_seconds=self.lock_ttl_seconds,
        )

        if not acquired:
            # Another region is already processing this provider_event_id.
            # Requeue with backoff instead of processing concurrently.
            self.queue.retry_later(
                task,
                error="global idempotency lock already held",
            )
            return False

        try:
            self._process_task(task)
            return True

        except TransientDatabaseError as exc:
            self.queue.retry_later(task, str(exc))
            return False

        except Exception as exc:
            self.queue.dead_letter(task, str(exc))
            return False

        finally:
            self.global_lock_store.release(
                key=lock_key,
                owner=lock_owner,
            )

    def _process_task(self, task: QueuedTask) -> None:
        event = task.event

        with self.uow_factory() as uow:
            existing = uow.webhook_events.find_by_provider_event_id(
                event.provider_event_id
            )

            if existing is not None:
                return

            uow.webhook_events.insert_event(event)

            self._process_business_logic(event, uow)

            uow.webhook_events.update_status(
                event.provider_event_id,
                WebhookStatus.PROCESSED,
            )

    def _process_business_logic(
        self,
        event: WebhookEvent,
        uow: UnitOfWork,
    ) -> None:
        if event.event_type != "payment.succeeded":
            return

        amount = float(event.payload["amount"])
        currency = event.payload.get("currency", "USD")

        uow.payments.insert_payment_impact(
            provider_event_id=event.provider_event_id,
            provider=event.provider,
            amount=amount,
            currency=currency,
        )

    def _lock_key(self, event: WebhookEvent) -> str:
        return f"webhook:{event.provider}:{event.provider_event_id}"

    def _lock_owner(self, task: QueuedTask) -> str:
        return f"{self.region_id}:{task.task_id}:{uuid.uuid4()}"


# ============================================================
# Sanity Test
# ============================================================

if __name__ == "__main__":
    db = InMemoryDatabase()
    global_lock = InMemoryGlobalLockStore()

    queue_region_1 = InMemoryTaskQueue(max_attempts=5, base_backoff_seconds=0.01)
    queue_region_2 = InMemoryTaskQueue(max_attempts=5, base_backoff_seconds=0.01)

    consumer_region_1 = WebhookConsumer(
        queue=queue_region_1,
        uow_factory=lambda: InMemoryUnitOfWork(db),
        global_lock_store=global_lock,
        lock_ttl_seconds=60,
        region_id="us-east-1",
    )

    consumer_region_2 = WebhookConsumer(
        queue=queue_region_2,
        uow_factory=lambda: InMemoryUnitOfWork(db),
        global_lock_store=global_lock,
        lock_ttl_seconds=60,
        region_id="eu-west-1",
    )

    event = WebhookEvent(
        provider_event_id="stripe_evt_123",
        provider="stripe",
        event_type="payment.succeeded",
        payload={"amount": 100.0, "currency": "USD"},
    )

    queue_region_1.publish(event)
    queue_region_2.publish(event)

    consumer_region_1.process_next()
    consumer_region_2.process_next()

    assert len(db.webhook_events) == 1
    assert len(db.payment_impacts) == 1

    print("Multi-region sanity test passed.")