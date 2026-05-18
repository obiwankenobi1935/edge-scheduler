"""FastAPI application: endpoints for gossip, invocation, scoring, and debugging.

M5 additions:
  - create_app now accepts SchedulerStats and DecisionLog
  - pick_node calls now record stats + decisions
  - GET /jobs/{job_id} fans out to peers when the job isn't found locally
  - GET /scheduler/stats   — running counters
  - GET /scheduler/decisions — last-N scheduling decisions
  - POST /scheduler/policy  — switch named policy preset
"""

from __future__ import annotations

import asyncio

import httpx
import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from edge_scheduler.jobs import JobStore
from edge_scheduler.metrics import MetricsSampler
from edge_scheduler.models import ExecuteRequest, InvokeRequest, InvokeResponse, PeerUpdate
from edge_scheduler.peers import PeerTable
from edge_scheduler.scheduler import pick_node
from edge_scheduler.scoring import ConfigStore, Policy, ScoringConfig
from edge_scheduler.stats import DecisionLog, SchedulerStats

log = structlog.get_logger()

REQUEST_TIMEOUT_S = 10.0


def create_app(
    table: PeerTable,
    jobs: JobStore,
    sampler: MetricsSampler,
    config_store: ConfigStore,      # M4
    node_id: str,
    stats: SchedulerStats | None = None,         # M5
    decision_log: DecisionLog | None = None,     # M5
    self_priority: int = 5,                      # M6
) -> FastAPI:
    app = FastAPI(title=f"edge-scheduler node {node_id}")

    # ------------------------------------------------------------------
    # M1 endpoints: gossip + debug
    # ------------------------------------------------------------------

    @app.post("/peer/update")
    async def peer_update(body: PeerUpdate) -> dict:
        table.update_last_seen(
            body.from_node_id,
            queue_depth=body.queue_depth,
            metrics=body.metrics,
            priority=body.priority,            # M6
        )
        # M4: adopt gossiped config if it is newer than ours.
        if body.scoring_config is not None:
            config_store.merge_gossip(body.scoring_config)
        return {"ok": True}

    @app.get("/peers")
    async def peers() -> JSONResponse:
        return JSONResponse(content=table.snapshot())

    @app.get("/health")
    async def health() -> dict:
        return {"node_id": node_id, "status": "ok"}

    # ------------------------------------------------------------------
    # M3 endpoint: local metrics
    # ------------------------------------------------------------------

    @app.get("/metrics")
    async def metrics() -> JSONResponse:
        return JSONResponse(content=sampler.latest.to_dict())

    # ------------------------------------------------------------------
    # M4 endpoints: scoring config inspection + hot-reload
    # ------------------------------------------------------------------

    @app.get("/scheduler/config")
    async def get_scheduler_config() -> JSONResponse:
        return JSONResponse(content=config_store.current.to_dict())

    @app.post("/scheduler/config")
    async def post_scheduler_config(body: ScoringConfig) -> JSONResponse:
        """Hot-reload weights. Change propagates to all peers within 2 seconds."""
        config_store.update_local(body)
        log.info("scoring_config_hot_reloaded", node=node_id)
        return JSONResponse(content=config_store.current.to_dict())

    # ------------------------------------------------------------------
    # M2 endpoints: invocation
    # ------------------------------------------------------------------

    @app.post("/invoke", response_model=InvokeResponse)
    async def invoke(body: InvokeRequest) -> InvokeResponse:
        chosen, forwarded = pick_node(
            self_node_id=node_id,
            self_queue_depth=jobs.queue_depth,
            self_metrics=sampler.latest,
            table=table,
            config_store=config_store,
            stats=stats,                    # M5
            decision_log=decision_log,      # M5
            self_priority=self_priority,    # M6
        )

        if not forwarded:
            job = jobs.create(
                function=body.function,
                args=body.args,
                scheduled_on=node_id,
                origin_node=node_id,
            )
            asyncio.create_task(jobs.run(job))
            log.info("invoke_local", job_id=job.job_id, function=body.function)
            return InvokeResponse(job_id=job.job_id, scheduled_on=node_id, forwarded=False)

        peer = table.get(chosen)
        if peer is None:
            raise HTTPException(status_code=503, detail=f"Peer {chosen} not found")

        job = jobs.create(
            function=body.function,
            args=body.args,
            scheduled_on=chosen,
            origin_node=node_id,
        )

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://{peer.host}:{peer.port}/execute",
                    json=ExecuteRequest(
                        job_id=job.job_id,
                        function=body.function,
                        args=body.args,
                        origin_node=node_id,
                    ).model_dump(),
                    timeout=REQUEST_TIMEOUT_S,
                )
                resp.raise_for_status()
        except Exception as exc:
            log.warning("forward_failed", peer=chosen, error=str(exc), job_id=job.job_id)
            job.scheduled_on = node_id
            asyncio.create_task(jobs.run(job))
            return InvokeResponse(job_id=job.job_id, scheduled_on=node_id, forwarded=False)

        log.info("invoke_forwarded", job_id=job.job_id, function=body.function, peer=chosen)
        return InvokeResponse(job_id=job.job_id, scheduled_on=chosen, forwarded=True)

    @app.post("/execute")
    async def execute(body: ExecuteRequest) -> dict:
        job = jobs.create(
            function=body.function,
            args=body.args,
            scheduled_on=node_id,
            origin_node=body.origin_node,
            job_id=body.job_id,
        )
        asyncio.create_task(jobs.run(job))
        log.info("execute_received", job_id=job.job_id, function=body.function,
                 from_node=body.origin_node)
        return {"ok": True, "job_id": job.job_id}

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str) -> JSONResponse:
        # Check locally first.
        job = jobs.get(job_id)
        if job is not None:
            return JSONResponse(content=job.to_dict())

        # M5: fan out to alive peers in parallel.
        alive_peers = [p for p in table.all().values() if p.status() == "alive"]

        async def _fetch_from_peer(peer) -> dict | None:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"http://{peer.host}:{peer.port}/jobs/{job_id}",
                        timeout=REQUEST_TIMEOUT_S,
                    )
                    if resp.status_code == 200:
                        return resp.json()
            except Exception:
                pass
            return None

        results = await asyncio.gather(*[_fetch_from_peer(p) for p in alive_peers])
        for result in results:
            if result is not None:
                return JSONResponse(content=result)

        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    @app.get("/queue")
    async def queue() -> JSONResponse:
        return JSONResponse(content={
            "node_id": node_id,
            "queue_depth": jobs.queue_depth,
            "jobs": [j.to_dict() for j in jobs.all()],
        })

    # ------------------------------------------------------------------
    # M5 endpoints: stats, decision log, policy switching
    # ------------------------------------------------------------------

    @app.get("/scheduler/stats")
    async def scheduler_stats() -> JSONResponse:
        """Running counters: invocations, local vs forwarded, avg cost score."""
        if stats is None:
            return JSONResponse(content={"detail": "stats not enabled"}, status_code=501)
        return JSONResponse(content=stats.to_dict())

    @app.get("/scheduler/decisions")
    async def scheduler_decisions() -> JSONResponse:
        """Last-50 scheduling decisions with candidate costs and chosen node."""
        if decision_log is None:
            return JSONResponse(content={"detail": "decision log not enabled"}, status_code=501)
        return JSONResponse(content=decision_log.to_list())

    @app.post("/scheduler/policy")
    async def set_policy(body: dict) -> JSONResponse:
        """Switch to a named policy preset (balanced/latency_first/energy_first/local_first).

        Body: {"policy": "<name>"}
        The new config is picked up by gossip within 2 s and propagates to all peers.
        """
        policy_name = body.get("policy")
        valid: list[Policy] = ["balanced", "latency_first", "energy_first", "local_first"]
        if policy_name not in valid:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown policy '{policy_name}'. Valid options: {valid}",
            )
        config_store.set_policy(policy_name)          # type: ignore[arg-type]
        log.info("policy_set_via_api", policy=policy_name, node=node_id)
        return JSONResponse(content=config_store.current.to_dict())

    return app
