from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, MutableMapping
from uuid import UUID

log = logging.getLogger(__name__)


class Span:
    """Thin wrapper over an optional OpenTelemetry span."""

    __slots__ = ("name", "attributes", "_otel_span", "_ended", "_owns_end")

    def __init__(
        self,
        name: str,
        *,
        otel_span: Any = None,
        attributes: dict[str, Any] | None = None,
        owns_end: bool = True,
    ) -> None:
        self.name = name
        self._otel_span = otel_span
        self.attributes: dict[str, Any] = dict(attributes or {})
        self._ended = False
        self._owns_end = owns_end

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value
        if self._otel_span is not None:
            try:
                self._otel_span.set_attribute(key, value)
            except Exception:
                pass

    def add_event(self, name: str, payload: dict[str, Any] | None = None) -> None:
        if self._otel_span is not None:
            try:
                self._otel_span.add_event(name, payload or {})
            except Exception:
                pass

    def record_exception(self, exc: BaseException) -> None:
        if self._otel_span is not None:
            try:
                self._otel_span.record_exception(exc)
            except Exception:
                pass

    def end(self) -> None:
        if self._ended:
            return
        self._ended = True
        if self._otel_span is not None and self._owns_end:
            try:
                self._otel_span.end()
            except Exception:
                pass

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self.record_exception(exc)
        self.end()


class Tracer:
    """Tracing facade with optional OpenTelemetry context propagation.

    Correlation IDs remain searchable attributes, while actual parent/child
    relationships use OpenTelemetry's current context and W3C propagation.
    """

    def __init__(self, *, service_name: str = "capsule-brain") -> None:
        self.service_name = service_name
        self._otel_tracer: Any = None
        self._enabled = False
        self._spans_started = 0
        self._provider: Any = None

    @classmethod
    def otel(
        cls,
        *,
        service_name: str = "capsule-brain",
        provider: Any = None,
        exporter_endpoint: str | None = None,
    ) -> "Tracer":
        """Build an OpenTelemetry-backed tracer when dependencies exist.

        If ``exporter_endpoint`` is supplied, an SDK TracerProvider and OTLP
        HTTP exporter are configured locally. Otherwise the global provider is
        used. Missing optional dependencies degrade safely to a no-op tracer.
        """
        tracer = cls(service_name=service_name)
        try:
            from opentelemetry import trace  # type: ignore[import-not-found]

            if provider is None and exporter_endpoint:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-not-found]
                    OTLPSpanExporter,
                )
                from opentelemetry.sdk.resources import Resource  # type: ignore[import-not-found]
                from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]
                from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore[import-not-found]

                provider = TracerProvider(
                    resource=Resource.create({"service.name": service_name})
                )
                provider.add_span_processor(
                    BatchSpanProcessor(OTLPSpanExporter(endpoint=exporter_endpoint))
                )
            if provider is None:
                provider = trace.get_tracer_provider()
            tracer._provider = provider
            tracer._otel_tracer = provider.get_tracer(service_name)
            tracer._enabled = True
        except Exception:
            log.debug("OpenTelemetry not available/configured; tracing is no-op")
        return tracer

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _attributes(
        self,
        correlation_id: UUID | str | None,
        attributes: dict[str, Any] | None,
    ) -> dict[str, Any]:
        attrs = dict(attributes or {})
        if correlation_id is not None:
            attrs.setdefault("correlation_id", str(correlation_id))
        attrs.setdefault("service", self.service_name)
        return attrs

    def start_span(
        self,
        name: str,
        *,
        correlation_id: UUID | str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start a span without making it current.

        Prefer ``span()`` for normal instrumentation so nested operations form
        proper parent/child relationships.
        """
        self._spans_started += 1
        attrs = self._attributes(correlation_id, attributes)
        if self._otel_tracer is None:
            return Span(name, attributes=attrs)
        try:
            otel_span = self._otel_tracer.start_span(name)
        except Exception:
            return Span(name, attributes=attrs)
        span = Span(name, otel_span=otel_span, attributes=attrs)
        for key, value in attrs.items():
            span.set_attribute(key, value)
        return span

    def inject_context(
        self, carrier: MutableMapping[str, str] | None = None
    ) -> dict[str, str]:
        target: MutableMapping[str, str] = carrier if carrier is not None else {}
        try:
            from opentelemetry.propagate import inject  # type: ignore[import-not-found]

            inject(target)
        except Exception:
            pass
        return dict(target)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        correlation_id: UUID | str | None = None,
        attributes: dict[str, Any] | None = None,
        trace_context: Mapping[str, str] | None = None,
    ) -> Iterator[Span]:
        """Create a current span, optionally continuing a propagated context."""
        self._spans_started += 1
        attrs = self._attributes(correlation_id, attributes)
        if self._otel_tracer is None:
            # Route through start_span so lightweight/custom tracer subclasses
            # keep observing spans without requiring OpenTelemetry.
            s = self.start_span(
                name, correlation_id=correlation_id, attributes=attributes
            )
            # start_span() already increments the counter. span() owns the
            # public accounting, so cancel the earlier increment above.
            self._spans_started -= 1
            try:
                yield s
            except Exception as exc:
                s.record_exception(exc)
                raise
            finally:
                s.end()
            return

        parent_context = None
        if trace_context:
            try:
                from opentelemetry.propagate import extract  # type: ignore[import-not-found]

                parent_context = extract(dict(trace_context))
            except Exception:
                parent_context = None

        try:
            cm = self._otel_tracer.start_as_current_span(
                name, context=parent_context
            )
        except Exception:
            s = Span(name, attributes=attrs)
            try:
                yield s
            finally:
                s.end()
            return

        with cm as otel_span:
            s = Span(
                name,
                otel_span=otel_span,
                attributes=attrs,
                owns_end=False,
            )
            for key, value in attrs.items():
                s.set_attribute(key, value)
            try:
                yield s
            except Exception as exc:
                s.record_exception(exc)
                raise
            finally:
                s._ended = True

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "service_name": self.service_name,
            "spans_started": self._spans_started,
        }


_default_tracer = Tracer()


def get_default_tracer() -> Tracer:
    return _default_tracer


def set_default_tracer(tracer: Tracer) -> None:
    global _default_tracer
    _default_tracer = tracer
