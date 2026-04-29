from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Set
import time


# ============================================================
# Event Models
# ============================================================

@dataclass(frozen=True)
class DispatchEvent:
    event_id: str
    partner_id: str
    merchant_id: str
    event_timestamp: float
    payload: Dict[str, Any]


class EventStatus(str, Enum):
    QUEUED = "queued"
    SENT = "sent"
    DROPPED = "dropped"
    FAILED = "failed"
    DEAD_LETTERED = "dead_lettered"


@dataclass(frozen=True)
class FailedEventRecord:
    event: DispatchEvent
    reason: str
    failed_at: float


# ============================================================
# Partner Configuration
# ============================================================

@dataclass(frozen=True)
class PartnerConfig:
    partner_id: str

    rate_per_second: float
    burst_capacity: int

    max_batch_size: int

    max_partner_queue_size: int
    max_merchant_queue_size: int

    max_events_per_merchant_per_round: int = 1

    max_retry_attempts: int = 3
    retry_backoff_seconds: float = 1.0


# ============================================================
# Token Bucket
# ============================================================

@dataclass
class TokenBucketState:
    rate_per_second: float
    burst_capacity: int
    available_tokens: float
    last_refill_timestamp: float


class TokenBucket:
    """
    Per-partner token bucket.

    One token represents permission to send one outbound batch request.
    This can be extended later if a partner charges tokens by event count
    instead of by request count.
    """

    def __init__(self, state: TokenBucketState) -> None:
        self.state = state

    def refill(self, now: float) -> None:
        """
        Refill tokens based on elapsed wall-clock time.

        Important detail:
        We cap tokens at burst_capacity so an idle partner can burst up to
        its configured limit, but never beyond it.
        """
        if self.state.last_refill_timestamp == 0.0:
            self.state.last_refill_timestamp = now
            return

        elapsed = max(0.0, now - self.state.last_refill_timestamp)
        refill_amount = elapsed * self.state.rate_per_second

        self.state.available_tokens = min(
            float(self.state.burst_capacity),
            self.state.available_tokens + refill_amount,
        )
        self.state.last_refill_timestamp = now

    def try_consume(self, now: float, amount: float = 1.0) -> bool:
        self.refill(now)

        if self.state.available_tokens >= amount:
            self.state.available_tokens -= amount
            return True

        return False


# ============================================================
# Per-Merchant and Per-Partner State
# ============================================================

@dataclass
class MerchantQueueState:
    merchant_id: str
    events: Deque[DispatchEvent] = field(default_factory=deque)

    total_enqueued: int = 0
    total_dropped: int = 0
    total_sent: int = 0
    total_failed: int = 0


@dataclass
class PartnerRuntimeState:
    config: PartnerConfig
    rate_limiter: TokenBucketState

    merchant_queues: Dict[str, MerchantQueueState] = field(default_factory=dict)

    active_merchants: Deque[str] = field(default_factory=deque)
    active_merchant_set: Set[str] = field(default_factory=set)

    total_buffered_events: int = 0

    total_sent_events: int = 0
    total_dropped_events: int = 0
    total_failed_events: int = 0

    dead_letter_queue: List[FailedEventRecord] = field(default_factory=list)


# ============================================================
# Dispatch Result Models
# ============================================================

@dataclass(frozen=True)
class BatchDispatchRequest:
    partner_id: str
    events: List[DispatchEvent]
    created_at: float


@dataclass(frozen=True)
class BatchDispatchResult:
    partner_id: str
    attempted_count: int
    succeeded_count: int
    failed_count: int
    failed_events: List[FailedEventRecord] = field(default_factory=list)


@dataclass
class DispatcherState:
    partners: Dict[str, PartnerRuntimeState] = field(default_factory=dict)

    # Lightweight idempotency guard for inbound duplicate events.
    processed_event_ids: Set[str] = field(default_factory=set)


# ============================================================
# Dispatcher Implementation
# ============================================================

class MultiTenantRateLimitedDispatcher:
    """
    Multi-tenant rate-limited dispatcher.

    Main guarantees:
      - partner-specific rate limits
      - partner-specific queue bounds
      - merchant-specific queue bounds
      - fair merchant scheduling within each partner
      - graceful send failure handling
      - duplicate event filtering
    """

    def __init__(
        self,
        partner_configs: List[PartnerConfig],
        send_batch_fn: Callable[[str, List[DispatchEvent]], None],
        state: Optional[DispatcherState] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.state = state or DispatcherState()
        self.send_batch_fn = send_batch_fn
        self.clock = clock

        now = self.clock()

        for config in partner_configs:
            self.state.partners[config.partner_id] = PartnerRuntimeState(
                config=config,
                rate_limiter=TokenBucketState(
                    rate_per_second=config.rate_per_second,
                    burst_capacity=config.burst_capacity,
                    available_tokens=float(config.burst_capacity),
                    last_refill_timestamp=now,
                ),
            )

    # ========================================================
    # Public API
    # ========================================================

    def submit_event(self, event: DispatchEvent) -> bool:
        """
        Submit an event into partner/merchant queue.

        Returns:
            True if accepted.
            False if rejected due to:
              - unknown partner
              - duplicate event
              - partner queue full
              - merchant queue full
        """
        partner_state = self.state.partners.get(event.partner_id)
        if partner_state is None:
            return False

        if event.event_id in self.state.processed_event_ids:
            return False

        merchant_state = self._get_or_create_merchant_state(
            partner_state,
            event.merchant_id,
        )

        if partner_state.total_buffered_events >= partner_state.config.max_partner_queue_size:
            self._record_drop(partner_state, merchant_state)
            return False

        if len(merchant_state.events) >= partner_state.config.max_merchant_queue_size:
            self._record_drop(partner_state, merchant_state)
            return False

        was_empty = len(merchant_state.events) == 0

        merchant_state.events.append(event)
        merchant_state.total_enqueued += 1
        partner_state.total_buffered_events += 1

        self.state.processed_event_ids.add(event.event_id)

        # Complex part:
        # active_merchants is the scheduling queue. A merchant appears at most
        # once in that queue. If we allowed duplicates, a whale merchant that
        # enqueues many events would appear many times and break fairness.
        if was_empty and event.merchant_id not in partner_state.active_merchant_set:
            partner_state.active_merchants.append(event.merchant_id)
            partner_state.active_merchant_set.add(event.merchant_id)

        return True

    def flush_once(self) -> Dict[str, BatchDispatchResult]:
        """
        Attempt one flush cycle across all partners.

        Semantics:
          - Each partner can send at most one batch per flush_once() call.
          - One rate-limit token allows one outbound batch request.
          - A failure for one partner does not stop other partners.
          - Partners with no work or no available token return a zero result.

        This matches the intended scheduler model:
            scheduler calls flush_once() repeatedly
            each call gives every partner one fair chance to send one batch
        """
        results: Dict[str, BatchDispatchResult] = {}
        now = self.clock()

        for partner_id, partner_state in self.state.partners.items():
            results[partner_id] = BatchDispatchResult(
                partner_id=partner_id,
                attempted_count=0,
                succeeded_count=0,
                failed_count=0,
            )

            if partner_state.total_buffered_events == 0:
                continue

            limiter = TokenBucket(partner_state.rate_limiter)

            # One token == one outbound partner batch request.
            if not limiter.try_consume(now, amount=1.0):
                continue

            batch = self.build_fair_batch(partner_state)

            if batch is None or not batch.events:
                continue

            # _send_batch_safely catches partner send failures and converts
            # them into a BatchDispatchResult, so other partners still proceed.
            results[partner_id] = self._send_batch_safely(
                partner_state=partner_state,
                batch=batch,
            )

        return results

    def build_fair_batch(
        self,
        partner_state: PartnerRuntimeState,
    ) -> Optional[BatchDispatchRequest]:
        """
        Build a fair batch for one partner using merchant round-robin.

        Complex part:
        We visit at most the merchants that were active at the start of this
        scheduling round. That prevents infinite loops when merchants still
        have remaining events and get re-added to the end of the queue.

        Fairness rule:
          - rotate across active merchants
          - each merchant contributes at most
            max_events_per_merchant_per_round
          - stop once max_batch_size is reached
        """
        if not partner_state.active_merchants:
            return None

        batch_events: List[DispatchEvent] = []
        merchants_to_visit = len(partner_state.active_merchants)

        while (
            merchants_to_visit > 0
            and len(batch_events) < partner_state.config.max_batch_size
            and partner_state.active_merchants
        ):
            merchant_id = partner_state.active_merchants.popleft()
            partner_state.active_merchant_set.remove(merchant_id)

            merchant_state = partner_state.merchant_queues.get(merchant_id)
            if merchant_state is None or not merchant_state.events:
                merchants_to_visit -= 1
                continue

            events_taken = 0

            while (
                merchant_state.events
                and len(batch_events) < partner_state.config.max_batch_size
                and events_taken < partner_state.config.max_events_per_merchant_per_round
            ):
                event = merchant_state.events.popleft()
                batch_events.append(event)
                partner_state.total_buffered_events -= 1
                events_taken += 1

            # If merchant still has pending work, put it at the back of the
            # active queue so other merchants get a turn first.
            if merchant_state.events:
                partner_state.active_merchants.append(merchant_id)
                partner_state.active_merchant_set.add(merchant_id)
            else:
                # Memory optimization:
                # Delete empty merchant queues. This prevents unbounded growth
                # if many one-off merchants appear over time.
                del partner_state.merchant_queues[merchant_id]

            merchants_to_visit -= 1

        if not batch_events:
            return None

        return BatchDispatchRequest(
            partner_id=partner_state.config.partner_id,
            events=batch_events,
            created_at=self.clock(),
        )

    def get_partner_metrics(self, partner_id: str) -> Dict[str, Any]:
        partner_state = self.state.partners.get(partner_id)
        if partner_state is None:
            return {}

        return {
            "partner_id": partner_id,
            "buffered_events": partner_state.total_buffered_events,
            "active_merchants": len(partner_state.active_merchants),
            "merchant_queues": len(partner_state.merchant_queues),
            "available_tokens": round(partner_state.rate_limiter.available_tokens, 3),
            "total_sent_events": partner_state.total_sent_events,
            "total_dropped_events": partner_state.total_dropped_events,
            "total_failed_events": partner_state.total_failed_events,
            "dlq_size": len(partner_state.dead_letter_queue),
        }

    # ========================================================
    # Failure Handling
    # ========================================================

    def _send_batch_safely(
        self,
        partner_state: PartnerRuntimeState,
        batch: BatchDispatchRequest,
    ) -> BatchDispatchResult:
        """
        Sends a batch and handles failures without corrupting dispatcher state.

        Current policy:
          - batch events are removed from queues before sending
          - if send succeeds, mark events as sent
          - if send fails, put failed records into partner DLQ

        Production alternative:
          - keep events in an in-flight state
          - acknowledge only after successful send
          - retry before DLQ
          - persist state in Kafka/SQS/Redis/RocksDB
        """
        try:
            self.send_batch_fn(batch.partner_id, batch.events)

            for event in batch.events:
                merchant_state = partner_state.merchant_queues.get(event.merchant_id)
                if merchant_state:
                    merchant_state.total_sent += 1

            partner_state.total_sent_events += len(batch.events)

            return BatchDispatchResult(
                partner_id=batch.partner_id,
                attempted_count=len(batch.events),
                succeeded_count=len(batch.events),
                failed_count=0,
            )

        except Exception as exc:
            now = self.clock()
            failed_records = [
                FailedEventRecord(
                    event=event,
                    reason=f"send failure: {exc}",
                    failed_at=now,
                )
                for event in batch.events
            ]

            partner_state.dead_letter_queue.extend(failed_records)
            partner_state.total_failed_events += len(batch.events)

            # Merchant-level failure metrics are best-effort because empty
            # merchant queues may already have been cleaned up after batching.
            for record in failed_records:
                merchant_state = partner_state.merchant_queues.get(record.event.merchant_id)
                if merchant_state:
                    merchant_state.total_failed += 1

            return BatchDispatchResult(
                partner_id=batch.partner_id,
                attempted_count=len(batch.events),
                succeeded_count=0,
                failed_count=len(batch.events),
                failed_events=failed_records,
            )

    # ========================================================
    # Helpers
    # ========================================================

    def _get_or_create_merchant_state(
        self,
        partner_state: PartnerRuntimeState,
        merchant_id: str,
    ) -> MerchantQueueState:
        merchant_state = partner_state.merchant_queues.get(merchant_id)

        if merchant_state is None:
            merchant_state = MerchantQueueState(merchant_id=merchant_id)
            partner_state.merchant_queues[merchant_id] = merchant_state

        return merchant_state

    def _record_drop(
        self,
        partner_state: PartnerRuntimeState,
        merchant_state: MerchantQueueState,
    ) -> None:
        partner_state.total_dropped_events += 1
        merchant_state.total_dropped += 1


# ============================================================
# Sanity Test
# ============================================================

if __name__ == "__main__":
    sent_batches: List[tuple[str, List[str], List[str]]] = []

    def fake_send(partner_id: str, events: List[DispatchEvent]) -> None:
        sent_batches.append(
            (
                partner_id,
                [event.event_id for event in events],
                [event.merchant_id for event in events],
            )
        )

    configs = [
        PartnerConfig(
            partner_id="google",
            rate_per_second=10.0,
            burst_capacity=2,
            max_batch_size=3,
            max_partner_queue_size=10,
            max_merchant_queue_size=5,
            max_events_per_merchant_per_round=1,
        ),
        PartnerConfig(
            partner_id="meta",
            rate_per_second=1.0,
            burst_capacity=1,
            max_batch_size=2,
            max_partner_queue_size=5,
            max_merchant_queue_size=3,
            max_events_per_merchant_per_round=1,
        ),
    ]

    dispatcher = MultiTenantRateLimitedDispatcher(
        partner_configs=configs,
        send_batch_fn=fake_send,
    )

    now = time.monotonic()

    # Whale merchant m1 submits many events.
    dispatcher.submit_event(
        DispatchEvent("e1", "google", "m1", now, {"x": 1})
    )
    dispatcher.submit_event(
        DispatchEvent("e2", "google", "m1", now, {"x": 2})
    )
    dispatcher.submit_event(
        DispatchEvent("e3", "google", "m1", now, {"x": 3})
    )

    # Smaller merchants also submit events.
    dispatcher.submit_event(
        DispatchEvent("e4", "google", "m2", now, {"x": 4})
    )
    dispatcher.submit_event(
        DispatchEvent("e5", "google", "m3", now, {"x": 5})
    )

    results = dispatcher.flush_once()

    assert results["google"].attempted_count == 3
    assert results["google"].succeeded_count == 3

    partner_id, event_ids, merchant_ids = sent_batches[0]

    assert partner_id == "google"

    # Fairness expectation:
    # one event from m1, one from m2, one from m3.
    assert set(merchant_ids) == {"m1", "m2", "m3"}

    # Duplicate should be rejected.
    duplicate_accepted = dispatcher.submit_event(
        DispatchEvent("e1", "google", "m9", now, {"x": 999})
    )
    assert duplicate_accepted is False

    print("Sanity test passed.")