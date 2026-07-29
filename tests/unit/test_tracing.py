import pytest
from uuid import uuid4

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.events.models import EventEnvelope
from capsule_brain.observability.tracing import (
    Span,
    Tracer,
    get_default_tracer,
    set_default_tracer,
)


def test_default_tracer_is_noop():
    tracer = Tracer()
    assert not tracer.enabled
    span = tracer.start_span("test")
    assert isinstance(span, Span)
    # No-op spans must not raise on any operation.
    span.set_attribute("k", "v")
    span.add_event("evt", {"x": 1})
    span.record_exception(ValueError("x"))
    span.end()
    # Idempotent end.
    span.end()


def test_span_context_manager_ends_on_exit():
    tracer = Tracer()
    with tracer.span("op") as span:
        span.set_attribute("a", 1)
    assert span._ended


def test_span_records_exception_on_raise():
    tracer = Tracer()
    with pytest.raises(ValueError, match="boom"):
        with tracer.span("op") as span:
            raise ValueError("boom")
    assert span._ended


def test_tracer_stats_count_spans():
    tracer = Tracer()
    tracer.start_span("a").end()
    with tracer.span("b"):
        pass
    stats = tracer.stats()
    assert stats["spans_started"] == 2
    assert stats["enabled"] is False


def test_otel_factory_degrades_gracefully_without_opentelemetry():
    # Even if opentelemetry is not installed, Tracer.otel() must not raise.
    tracer = Tracer.otel(service_name="test")
    assert isinstance(tracer, Tracer)
    # Span creation still works (as no-op).
    span = tracer.start_span("x")
    span.end()


def test_correlation_id_attached_to_span():
    tracer = Tracer()
    cid = uuid4()
    span = tracer.start_span("op", correlation_id=cid)
    assert span.attributes["correlation_id"] == str(cid)
    assert span.attributes["service"] == "capsule-brain"
    span.end()


@pytest.mark.asyncio
async def test_event_bus_publish_creates_span():
    """Publishing an event should create a span with the event's correlation id."""
    bus = LocalEventBus()
    await bus.start()

    captured = []

    class CapturingTracer(Tracer):
        def start_span(self, name, *, correlation_id=None, attributes=None):
            span = super().start_span(name, correlation_id=correlation_id, attributes=attributes)
            captured.append((name, span.attributes))
            return span

    set_default_tracer(CapturingTracer())
    try:
        cid = uuid4()
        await bus.publish(
            EventEnvelope(
                event_type="test.event",
                source="test",
                correlation_id=cid,
                payload={},
            )
        )
        assert any(name.startswith("event.publish") for name, _ in captured)
        # The span attributes should carry the correlation id.
        found = False
        for _, attrs in captured:
            if attrs.get("correlation_id") == str(cid):
                found = True
                break
        assert found
    finally:
        set_default_tracer(Tracer())
        await bus.stop()


@pytest.mark.asyncio
async def test_event_bus_publish_traces_handler_failures():
    bus = LocalEventBus()
    await bus.start()

    def bad_handler(event):
        raise RuntimeError("handler exploded")

    bus.subscribe("test.event", bad_handler)

    class CapturingSpan(Span):
        # Subclass to bypass __slots__ restriction on monkey-patching.
        def add_event(self, name, payload=None):
            tracer.events.append((name, payload))
            super().add_event(name, payload or {})

    class CountingTracer(Tracer):
        events: list

        def __init__(self):
            super().__init__()
            self.events = []

        def start_span(self, name, *, correlation_id=None, attributes=None):
            attrs = dict(attributes or {})
            if correlation_id is not None:
                attrs.setdefault("correlation_id", str(correlation_id))
            attrs.setdefault("service", self.service_name)
            return CapturingSpan(name, attributes=attrs)

    tracer = CountingTracer()
    set_default_tracer(tracer)
    try:
        await bus.publish(
            EventEnvelope(
                event_type="test.event",
                source="test",
                payload={},
            )
        )
        assert any(name == "handler_failed" for name, _ in tracer.events)
    finally:
        set_default_tracer(Tracer())
        await bus.stop()


def test_otel_span_uses_current_context_manager():
    class FakeOTelSpan:
        def __init__(self):
            self.attributes = {}
            self.events = []

        def set_attribute(self, key, value):
            self.attributes[key] = value

        def add_event(self, name, payload):
            self.events.append((name, payload))

        def record_exception(self, _exc):
            pass

    class FakeSpanContextManager:
        def __init__(self, owner, name, context):
            self.owner = owner
            self.name = name
            self.context = context
            self.span = FakeOTelSpan()

        def __enter__(self):
            self.owner.entered += 1
            self.owner.calls.append((self.name, self.context))
            return self.span

        def __exit__(self, exc_type, exc, tb):
            self.owner.exited += 1

    class FakeOTelTracer:
        def __init__(self):
            self.calls = []
            self.entered = 0
            self.exited = 0

        def start_as_current_span(self, name, *, context=None):
            return FakeSpanContextManager(self, name, context)

        def start_span(self, _name):
            raise AssertionError("span() must use start_as_current_span")

    class FakeProvider:
        def __init__(self):
            self.tracer = FakeOTelTracer()

        def get_tracer(self, _service_name):
            return self.tracer

    provider = FakeProvider()
    tracer = Tracer.otel(service_name="hardening-test", provider=provider)
    with tracer.span("child", correlation_id="cid") as span:
        assert span.attributes["correlation_id"] == "cid"

    assert provider.tracer.entered == 1
    assert provider.tracer.exited == 1
    assert provider.tracer.calls[0][0] == "child"


def test_otel_sdk_nested_spans_form_parent_child_trace():
    trace_sdk = pytest.importorskip("opentelemetry.sdk.trace")
    exporter_mod = pytest.importorskip(
        "opentelemetry.sdk.trace.export.in_memory_span_exporter"
    )
    export_mod = pytest.importorskip("opentelemetry.sdk.trace.export")

    provider = trace_sdk.TracerProvider()
    exporter = exporter_mod.InMemorySpanExporter()
    provider.add_span_processor(export_mod.SimpleSpanProcessor(exporter))
    tracer = Tracer.otel(service_name="trace-test", provider=provider)

    with tracer.span("root"):
        with tracer.span("child"):
            pass

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert set(spans) == {"root", "child"}
    assert spans["child"].context.trace_id == spans["root"].context.trace_id
    assert spans["child"].parent is not None
    assert spans["child"].parent.span_id == spans["root"].context.span_id


@pytest.mark.asyncio
async def test_async_event_handler_continues_publish_trace():
    trace_sdk = pytest.importorskip("opentelemetry.sdk.trace")
    exporter_mod = pytest.importorskip(
        "opentelemetry.sdk.trace.export.in_memory_span_exporter"
    )
    export_mod = pytest.importorskip("opentelemetry.sdk.trace.export")
    from capsule_brain.runtime.tasks import TaskRegistry

    provider = trace_sdk.TracerProvider()
    exporter = exporter_mod.InMemorySpanExporter()
    provider.add_span_processor(export_mod.SimpleSpanProcessor(exporter))
    tracer = Tracer.otel(service_name="event-trace-test", provider=provider)
    set_default_tracer(tracer)

    bus = LocalEventBus({"async_dispatch": True})
    tasks = TaskRegistry()
    bus.set_task_registry(tasks)
    handled = __import__("asyncio").Event()

    async def handler(_event):
        handled.set()

    bus.subscribe("trace.async", handler)
    await bus.start()
    try:
        await bus.publish(
            EventEnvelope(event_type="trace.async", source="test", payload={})
        )
        await __import__("asyncio").wait_for(handled.wait(), timeout=1.0)
        # Let the managed handler span exit before inspecting finished spans.
        await __import__("asyncio").sleep(0)
        spans = {span.name: span for span in exporter.get_finished_spans()}
        publish = spans["event.publish:trace.async"]
        child = spans["event.handler:trace.async"]
        assert child.context.trace_id == publish.context.trace_id
        assert child.parent is not None
        assert child.parent.span_id == publish.context.span_id
    finally:
        await tasks.shutdown()
        await bus.stop()
        set_default_tracer(Tracer())
