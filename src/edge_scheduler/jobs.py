"""In-memory job store and background execution runner."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import structlog

from edge_scheduler.functions import get as get_function
from edge_scheduler.models import JobRecord

log = structlog.get_logger()


class JobStore:
    """Tracks all jobs on this node (both locally-run and received via /execute)."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._queue_depth: int = 0          # live count of pending + running jobs

    # ------------------------------------------------------------------
    # Queue depth (read by scheduler + gossip)
    # ------------------------------------------------------------------

    @property
    def queue_depth(self) -> int:
        return self._queue_depth

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    def create(
        self,
        function: str,
        args: dict,
        scheduled_on: str,
        origin_node: str,
        job_id: str | None = None,
    ) -> JobRecord:
        """Register a new job and return it."""
        job = JobRecord(
            job_id=job_id or str(uuid.uuid4()),
            function=function,
            args=args,
            status="pending",
            scheduled_on=scheduled_on,
            origin_node=origin_node,
            created_at=datetime.now(timezone.utc),
        )
        self._jobs[job.job_id] = job
        self._queue_depth += 1
        return job

    def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def all(self) -> list[JobRecord]:
        return list(self._jobs.values())

    # ------------------------------------------------------------------
    # Background execution
    # ------------------------------------------------------------------

    async def run(self, job: JobRecord) -> None:
        """Execute a job's function in a thread pool and update its record.

        We use asyncio.to_thread() so the CPU-bound work runs in a separate
        OS thread and doesn't block the event loop (which would freeze gossip
        and incoming HTTP requests for the duration of the computation).
        """
        job.status = "running"
        log.info("job_start", job_id=job.job_id, function=job.function, args=job.args)
        start = asyncio.get_event_loop().time()
        try:
            fn = get_function(job.function)
            # to_thread: runs fn(**args) in a ThreadPoolExecutor, returns when done.
            result = await asyncio.to_thread(fn, **job.args)
            job.status = "done"
            job.result = result
            log.info("job_done", job_id=job.job_id, duration_ms=round((asyncio.get_event_loop().time() - start) * 1000, 2))
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            log.warning("job_failed", job_id=job.job_id, error=str(exc))
        finally:
            job.finished_at = datetime.now(timezone.utc)
            job.duration_ms = round((asyncio.get_event_loop().time() - start) * 1000, 2)
            self._queue_depth = max(0, self._queue_depth - 1)
