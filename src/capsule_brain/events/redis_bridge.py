from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

try:
    import redis.asyncio as redis  # type: ignore[import-not-found]
except ImportError:  # optional cross-process dependency
    redis = None  # type: ignore[assignment]


from capsule_brain.runtime.service import CapsuleService, HealthStatus, ServiceState
from .local_bus import LocalEventBus
from .models import EventEnvelope

log = logging.getLogger(__name__)


class RedisBridge(CapsuleService):
    """Optional Redis bridge for cross-process events.

    Internal Capsule Brain services should use LocalEventBus directly.
    This bridge mirrors selected topics between Redis and the local event bus.
    """

    name = "redis_bridge"

    def __init__(
        self,
        local_bus: LocalEventBus,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(cfg)
        self.local_bus = local_bus
        self.redis_url = self.cfg.get("url", "redis://localhost:6379/0")
        self.channel_prefix = self.cfg.get("channel_prefix", "capsule")
        self.inbound_topics: set[str] = set(self.cfg.get("inbound_topics", []))
        self.outbound_topics: set[str] = set(self.cfg.get("outbound_topics", []))
        self._redis: Any = None
        self._pubsub = None
        self._listener_task: asyncio.Task | None = None
        self._unsubscribers: list[Callable[[], None]] = []
        self._messages_in = 0
        self._messages_out = 0
        self._errors = 0
        # Keep a bounded memory of event IDs received from Redis. This prevents
        # an inbound topic that is also configured outbound from being echoed
        # back to Redis indefinitely, including when LocalEventBus dispatches
        # handlers asynchronously after publish() returns.
        self._inbound_event_ids: OrderedDict[UUID, None] = OrderedDict()
        self._inbound_event_id_limit = max(128, int(self.cfg.get("dedupe_ids", 4096)))

    def _channel(self, topic: str) -> str:
        return f"{self.channel_prefix}:{topic}"

    def _remember_inbound(self, event_id: UUID) -> None:
        self._inbound_event_ids[event_id] = None
        self._inbound_event_ids.move_to_end(event_id)
        while len(self._inbound_event_ids) > self._inbound_event_id_limit:
            self._inbound_event_ids.popitem(last=False)

    def _encode_event(self, event: EventEnvelope) -> str:
        """Serialize an event without dropping correlation or trace context."""
        payload = {
            "event_type": event.event_type,
            "payload": event.payload,
            "source": event.source,
            "event_id": str(event.event_id),
            "correlation_id": (
                str(event.correlation_id) if event.correlation_id else None
            ),
            "created_at": event.created_at.isoformat(),
            "trace_context": dict(event.trace_context),
        }
        return json.dumps(payload)

    @staticmethod
    def _decode_event(topic: str, data: str) -> EventEnvelope:
        """Reconstruct an event received from Redis, preserving identity/context."""
        raw = json.loads(data)
        event_id = raw.get("event_id")
        correlation_id = raw.get("correlation_id")
        created_at = raw.get("created_at")
        return EventEnvelope(
            event_type=raw.get("event_type", topic),
            payload=raw.get("payload", {}),
            source=raw.get("source", "redis"),
            event_id=UUID(event_id) if event_id else uuid4(),
            correlation_id=UUID(correlation_id) if correlation_id else None,
            trace_context=dict(raw.get("trace_context") or {}),
            created_at=(
                datetime.fromisoformat(created_at)
                if created_at
                else datetime.now(timezone.utc)
            ),
        )

    async def start(self) -> None:
        self.state = ServiceState.STARTING
        if redis is None:
            raise RuntimeError(
                "RedisBridge requires the optional 'redis' dependency (redis>=5.0)"
            )
        # Mark degraded before scheduling the reconnect loop so the listener's
        # state guard cannot race startup and exit while state is STARTING.
        self.state = ServiceState.DEGRADED
        # The Redis connection is established inside _listen()'s reconnect
        # loop so that transient connection failures at startup don't kill
        # the bridge permanently.
        if self.inbound_topics or self.outbound_topics:
            self._listener_task = asyncio.create_task(
                self._listen(),
                name="redis-bridge-listener",
            )

        for topic in sorted(self.outbound_topics):
            self._unsubscribers.append(
                self.local_bus.subscribe(
                    topic,
                    self._make_outbound_handler(topic),
                )
            )

        # State remains DEGRADED until _listen() establishes the Redis
        # connection and sets it to RUNNING.

    def _make_outbound_handler(self, topic: str):
        async def handler(event: EventEnvelope) -> None:
            if event.event_id in self._inbound_event_ids:
                # This exact event arrived from Redis already. Do not mirror it
                # back out when inbound/outbound topic sets overlap.
                return
            if self._redis is None:
                self._errors += 1
                log.warning(
                    "Redis outbound publish skipped "
                    "(not connected) for %s",
                    topic,
                )
                return
            try:
                await self._redis.publish(
                    self._channel(topic), self._encode_event(event)
                )
                self._messages_out += 1
            except Exception:
                self._errors += 1
                log.exception("Redis outbound publish failed for %s", topic)

        return handler

    async def _listen(self) -> None:
        """Listen for inbound Redis messages with automatic reconnection.

        On connection loss (Redis restart, network blip), the listener
        enters an exponential backoff retry loop (1s → 2s → 4s → ... → 30s
        cap) and reconnects when the server becomes available again. This
        prevents a single transient failure from permanently killing
        cross-process messaging.
        """
        backoff = 1.0
        while self.state in {ServiceState.RUNNING, ServiceState.DEGRADED}:
            try:
                if self._redis is None:
                    self._redis = redis.from_url(
                        self.redis_url, decode_responses=True
                    )
                    await self._redis.ping()
                    self._pubsub = self._redis.pubsub(
                        ignore_subscribe_messages=True
                    )
                    if self.inbound_topics:
                        await self._pubsub.subscribe(
                            *(
                                self._channel(topic)
                                for topic in sorted(self.inbound_topics)
                            )
                        )
                    self.state = ServiceState.RUNNING
                    backoff = 1.0
                    log.info("Redis bridge connected.")

                if not self.inbound_topics:
                    # Outbound-only bridges still need a maintained connection.
                    await asyncio.sleep(1.0)
                    continue

                assert self._pubsub is not None
                async for message in self._pubsub.listen():
                    if self.state in {ServiceState.STOPPING, ServiceState.STOPPED}:
                        return
                    if message.get("type") != "message":
                        continue

                    try:
                        channel = message["channel"]
                        prefix = f"{self.channel_prefix}:"
                        if not channel.startswith(prefix):
                            continue
                        topic = channel[len(prefix):]

                        event = self._decode_event(topic, message["data"])
                        self._remember_inbound(event.event_id)
                        await self.local_bus.publish(event)
                        self._messages_in += 1
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        self._errors += 1
                        log.exception("Failed processing inbound Redis message")

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self.state in {ServiceState.STOPPING, ServiceState.STOPPED}:
                    return
                self._errors += 1
                self.state = ServiceState.DEGRADED
                log.warning(
                    "Redis bridge connection lost (%s). Retrying in %.1fs...",
                    exc,
                    backoff,
                )
                await self._cleanup_redis_clients()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)

    async def _cleanup_redis_clients(self) -> None:
        """Close and null out Redis clients after a connection failure."""
        if self._pubsub is not None:
            try:
                await self._pubsub.close()
            except Exception:
                pass
            self._pubsub = None
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception:
                pass
            self._redis = None

    async def stop(self) -> None:
        self.state = ServiceState.STOPPING

        for unsubscribe in self._unsubscribers:
            try:
                unsubscribe()
            except Exception:
                log.exception("Redis bridge unsubscribe failed")
        self._unsubscribers.clear()

        if self._listener_task:
            self._listener_task.cancel()
            await asyncio.gather(self._listener_task, return_exceptions=True)
            self._listener_task = None

        if self._pubsub is not None:
            await self._pubsub.close()
            self._pubsub = None

        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

        self.state = ServiceState.STOPPED

    async def health(self) -> HealthStatus:
        return HealthStatus(
            state=self.state,
            details={
                "messages_in": self._messages_in,
                "messages_out": self._messages_out,
                "errors": self._errors,
                "inbound_topics": sorted(self.inbound_topics),
                "outbound_topics": sorted(self.outbound_topics),
            },
        )
