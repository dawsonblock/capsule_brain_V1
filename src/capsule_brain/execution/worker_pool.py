from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from capsule_brain.observability.tracing import get_default_tracer
from capsule_brain.runtime.service import CapsuleService, HealthStatus, ServiceState

from .models import ExecutionRequest, ExecutionResult

log = logging.getLogger(__name__)


@dataclass(slots=True)
class _PendingJob:
    request: ExecutionRequest
    correlation_id: Any
    future: asyncio.Future[ExecutionResult]
    runner_task: asyncio.Task[ExecutionResult] | None = None
    enqueued_at: float = field(
        default_factory=lambda: asyncio.get_event_loop().time()
    )


class ExecutionWorkerPool(CapsuleService):
    """A bounded async worker pool for execution jobs.

    The pool wraps a runner (ExecutionRunner or ContainerExecutionRunner)
    and limits concurrent executions to ``max_workers``. Jobs are queued
    FIFO and dispatched to ``max_workers`` permanent worker tasks. This
    prevents Docker daemon CPU/RAM thrashing under heavy multi-task load
    and provides a natural extension point for running execution jobs on
    remote worker nodes (the runner interface is unchanged).

    The pool is a CapsuleService so it participates in the normal
    dependency-ordered lifecycle. On stop, in-flight jobs are awaited (with
    a configurable drain timeout) so partial results are not lost.
    """

    name = "execution_worker_pool"

    def __init__(
        self,
        runner: Any,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(cfg)
        self.runner = runner
        self.max_workers = max(1, int(self.cfg.get("max_workers", 2)))
        # 0 = unbounded queue. A positive value rejects submissions when the
        # queue is full, providing backpressure to upstream callers.
        self.queue_max = max(0, int(self.cfg.get("queue_max", 0)))
        self.drain_timeout_s = float(self.cfg.get("drain_timeout_s", 10.0))
        self._queue: asyncio.Queue[_PendingJob | None] = asyncio.Queue(
            maxsize=self.queue_max
        )
        self._workers: list[asyncio.Task] = []
        self._stop_event = asyncio.Event()
        # Metrics
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._cancelled = 0
        self._active = 0
        self._peak_active = 0
        self._queue_depth_high_water = 0

    async def start(self) -> None:
        self.state = ServiceState.STARTING
        self._stop_event.clear()
        # Spawn max_workers permanent consumer tasks. Each pulls jobs from
        # the queue and runs them through the wrapped runner. A None
        # sentinel signals a worker to exit cleanly on stop.
        loop = asyncio.get_event_loop()
        self._workers = [
            loop.create_task(self._worker_loop(i))
            for i in range(self.max_workers)
        ]
        self.state = ServiceState.RUNNING

    async def stop(self) -> None:
        self.state = ServiceState.STOPPING
        self._stop_event.set()
        # Cancel futures for any jobs still queued (callers awaiting them
        # get a CancelledError). In-flight jobs are allowed to finish during
        # the drain timeout below.
        while True:
            try:
                leftover = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                if leftover is not None and not leftover.future.done():
                    leftover.future.cancel()
            finally:
                self._queue.task_done()
        # Signal every worker to exit. Sentinels share the bounded queue with
        # jobs, so enqueue them asynchronously: with queue_max < max_workers,
        # put_nowait() would otherwise signal only the first few workers and
        # force the rest to be cancelled after the drain timeout.
        async def signal_workers() -> None:
            for _ in self._workers:
                await self._queue.put(None)

        signal_task: asyncio.Task[None] | None = None
        if self._workers:
            signal_task = asyncio.create_task(signal_workers())
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        signal_task,
                        *self._workers,
                        return_exceptions=True,
                    ),
                    timeout=self.drain_timeout_s,
                )
            except asyncio.TimeoutError:
                log.warning(
                    "ExecutionWorkerPool drain timed out after %.1fs; "
                    "cancelling %d workers",
                    self.drain_timeout_s,
                    len(self._workers),
                )
                if signal_task is not None and not signal_task.done():
                    signal_task.cancel()
                for worker in self._workers:
                    worker.cancel()
                await asyncio.gather(
                    *(
                        [signal_task] if signal_task is not None else []
                    ),
                    *self._workers,
                    return_exceptions=True,
                )
        self._workers.clear()
        self.state = ServiceState.STOPPED

    async def submit(
        self,
        request: ExecutionRequest,
        *,
        correlation_id: Any = None,
    ) -> ExecutionResult:
        """Submit an execution job and await its result.

        The caller is suspended until a worker picks up the job and the
        runner completes. This preserves the synchronous ``execute`` API of
        ExecutionService while bounding concurrency behind the scenes.
        """
        if self.state != ServiceState.RUNNING:
            raise RuntimeError("ExecutionWorkerPool is not running")
        loop = asyncio.get_running_loop()
        job = _PendingJob(
            request=request,
            correlation_id=correlation_id,
            future=loop.create_future(),
        )
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            raise RuntimeError(
                f"ExecutionWorkerPool queue is full ({self.queue_max})"
            ) from exc
        self._submitted += 1
        depth = self._queue.qsize()
        if depth > self._queue_depth_high_water:
            self._queue_depth_high_water = depth
        try:
            return await job.future
        except asyncio.CancelledError:
            # Prevent queued side effects from running after the caller has
            # abandoned the request. If already active, propagate cancellation
            # to the runner so host/container subprocesses are terminated.
            if not job.future.done():
                job.future.cancel()
            if job.runner_task is not None and not job.runner_task.done():
                job.runner_task.cancel()
            raise

    async def _worker_loop(self, worker_id: int) -> None:
        """Permanent consumer: pull jobs and run them until a None sentinel.

        Each worker consumes exactly one None sentinel and then exits. We do
        NOT drain remaining queued jobs here — stop() cancels the futures of
        queued jobs before sending sentinels, so no worker needs to process
        them during shutdown.
        """
        while True:
            job = await self._queue.get()
            if job is None:
                self._queue.task_done()
                return
            try:
                if job.future.cancelled():
                    self._cancelled += 1
                    continue
                await self._run_job(job)
            finally:
                self._queue.task_done()

    async def _run_job(self, job: _PendingJob) -> None:
        tracer = get_default_tracer()
        self._active += 1
        if self._active > self._peak_active:
            self._peak_active = self._active
        try:
            with tracer.span(
                "execution.worker.run",
                correlation_id=job.correlation_id,
                attributes={
                    "request_id": job.request.id,
                    "command": " ".join(str(c) for c in job.request.command),
                },
            ):
                job.runner_task = asyncio.create_task(self.runner.run(job.request))
                result = await job.runner_task
            self._completed += 1
            if not job.future.done():
                job.future.set_result(result)
        except asyncio.CancelledError:
            if job.runner_task is not None and not job.runner_task.done():
                job.runner_task.cancel()
                await asyncio.gather(job.runner_task, return_exceptions=True)
            if job.future.cancelled():
                self._cancelled += 1
                return
            if not job.future.done():
                job.future.cancel()
            raise
        except Exception as exc:
            self._failed += 1
            if not job.future.done():
                job.future.set_exception(exc)
        finally:
            job.runner_task = None
            self._active -= 1

    async def health(self) -> HealthStatus:
        state = (
            ServiceState.DEGRADED
            if self.state == ServiceState.RUNNING and self._failed > 0
            else self.state
        )
        return HealthStatus(
            state=state,
            details={
                "max_workers": self.max_workers,
                "queue_max": self.queue_max,
                "queue_depth": self._queue.qsize(),
                "queue_high_water": self._queue_depth_high_water,
                "active": self._active,
                "peak_active": self._peak_active,
                "submitted": self._submitted,
                "completed": self._completed,
                "failed": self._failed,
                "cancelled": self._cancelled,
                "runner": type(self.runner).__name__,
            },
        )
