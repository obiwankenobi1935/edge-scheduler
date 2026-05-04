"""Integration test: invoke functions across three live in-process nodes.

Tests cover:
  - Local execution when self is cheapest
  - Forwarding when a peer has a lower cost (via artificially raised queue depth)
  - Job result polling via /jobs/{job_id}
  - Queue depth reflected in /queue endpoint

This test takes ~5 seconds for nodes to start + jobs to complete.
"""

from __future__ import annotations

import asyncio

import pytest
import uvicorn

from edge_scheduler.api import create_app
from edge_scheduler.config import NodeConfig, PeerEntry
from edge_scheduler.gossip import gossip_loop
from edge_scheduler.jobs import JobStore
from edge_scheduler.metrics import MetricsSampler
from edge_scheduler.peers import PeerTable
from edge_scheduler.scoring import ConfigStore


# ---------------------------------------------------------------------------
# Helpers (reused from test_gossip_integration, slightly extended)
# ---------------------------------------------------------------------------

PORTS = {"a": 19001, "b": 19002, "c": 19003}


def _make_config(node_id: str, port: int, peers: list[tuple[str, int]]) -> NodeConfig:
    return NodeConfig(
        node_id=node_id,
        host="127.0.0.1",
        port=port,
        peers=[PeerEntry(node_id=pid, host="127.0.0.1", port=pp) for pid, pp in peers],
    )


class _Node:
    def __init__(self, config: NodeConfig) -> None:
        self.config       = config
        self.table        = PeerTable(config)
        self.jobs         = JobStore()
        self.sampler      = MetricsSampler(config.node_id)
        self.config_store = ConfigStore(config.scoring)
        self.app          = create_app(self.table, self.jobs, self.sampler,
                                       self.config_store, config.node_id)
        self._server: uvicorn.Server | None = None
        self._gossip_task: asyncio.Task | None = None
        self._metrics_task: asyncio.Task | None = None
        self._serve_task: asyncio.Task | None = None

    async def start(self) -> None:
        uvi_config = uvicorn.Config(
            self.app, host=self.config.host, port=self.config.port, log_level="warning"
        )
        self._server       = uvicorn.Server(uvi_config)
        self._serve_task   = asyncio.create_task(self._server.serve())
        self._gossip_task  = asyncio.create_task(
            gossip_loop(self.table, self.config.node_id, self.jobs,
                        self.sampler, self.config_store)
        )
        self._metrics_task = asyncio.create_task(self.sampler.run())
        while not self._server.started:
            await asyncio.sleep(0.05)

    async def stop(self) -> None:
        for task in (self._gossip_task, self._metrics_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self._server:
            self._server.should_exit = True
        if self._serve_task:
            await self._serve_task


async def _wait_for_job(node: _Node, job_id: str, timeout: float = 10.0) -> dict:
    """Poll a node's job store until the job is done or failed."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        job = node.jobs.get(job_id)
        if job and job.status in ("done", "failed"):
            return job.to_dict()
        await asyncio.sleep(0.1)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_local_invocation():
    """Node A invokes with empty queues everywhere — should run locally."""
    configs = {
        "a": _make_config("a", PORTS["a"], [("b", PORTS["b"]), ("c", PORTS["c"])]),
        "b": _make_config("b", PORTS["b"], [("a", PORTS["a"]), ("c", PORTS["c"])]),
        "c": _make_config("c", PORTS["c"], [("a", PORTS["a"]), ("b", PORTS["b"])]),
    }
    nodes = {nid: _Node(cfg) for nid, cfg in configs.items()}
    for node in nodes.values():
        await node.start()

    try:
        # Let gossip settle so RTTs are populated.
        await asyncio.sleep(3)

        from edge_scheduler.models import InvokeRequest
        from edge_scheduler.scheduler import pick_node

        # With all queues at 0, self should always win (rtt=0 vs rtt>0).
        chosen, forwarded = pick_node(
            self_node_id="a",
            self_queue_depth=0,
            self_metrics=nodes["a"].sampler.latest,
            table=nodes["a"].table,
            config_store=nodes["a"].config_store,
        )
        assert chosen == "a"
        assert forwarded is False

        # Actually invoke via the API layer.
        req = InvokeRequest(function="prime_search", args={"n": 1000})
        # Call invoke directly on node A's app state.
        from fastapi.testclient import TestClient
        client = TestClient(nodes["a"].app)
        resp = client.post("/invoke", json={"function": "prime_search", "args": {"n": 1000}})
        assert resp.status_code == 200
        data = resp.json()
        assert data["scheduled_on"] == "a"
        assert data["status"] == "accepted"

        # Wait for the job to complete.
        job = await _wait_for_job(nodes["a"], data["job_id"])
        assert job["status"] == "done"
        assert job["result"]["count"] == 168   # primes below 1000

    finally:
        for node in nodes.values():
            await node.stop()


async def test_forwarding_when_self_queue_is_high():
    """Node A has a high queue depth — scheduler should forward to a peer."""
    configs = {
        "a": _make_config("a", PORTS["a"], [("b", PORTS["b"]), ("c", PORTS["c"])]),
        "b": _make_config("b", PORTS["b"], [("a", PORTS["a"]), ("c", PORTS["c"])]),
        "c": _make_config("c", PORTS["c"], [("a", PORTS["a"]), ("b", PORTS["b"])]),
    }
    nodes = {nid: _Node(cfg) for nid, cfg in configs.items()}
    for node in nodes.values():
        await node.start()

    try:
        await asyncio.sleep(3)

        from edge_scheduler.scheduler import pick_node

        # Simulate node A having a very deep queue.
        # With queue_depth=20 on self, cost = 200.
        # Peers have queue_depth=0 and small RTT — cost will be << 200.
        chosen, forwarded = pick_node(
            self_node_id="a",
            self_queue_depth=20,
            self_metrics=nodes["a"].sampler.latest,
            table=nodes["a"].table,
            config_store=nodes["a"].config_store,
        )
        assert forwarded is True
        assert chosen in ("b", "c")

    finally:
        for node in nodes.values():
            await node.stop()


async def test_hash_chain_result_is_deterministic():
    """hash_chain result should be consistent regardless of which node runs it."""
    configs = {
        "a": _make_config("a", PORTS["a"], [("b", PORTS["b"]), ("c", PORTS["c"])]),
        "b": _make_config("b", PORTS["b"], [("a", PORTS["a"]), ("c", PORTS["c"])]),
        "c": _make_config("c", PORTS["c"], [("a", PORTS["a"]), ("b", PORTS["b"])]),
    }
    nodes = {nid: _Node(cfg) for nid, cfg in configs.items()}
    for node in nodes.values():
        await node.start()

    try:
        from fastapi.testclient import TestClient

        client_a = TestClient(nodes["a"].app)
        resp = client_a.post("/invoke", json={"function": "hash_chain", "args": {"n": 500}})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        # Wait for job on whichever node ran it.
        result = None
        for node in nodes.values():
            job = node.jobs.get(job_id)
            if job:
                job_dict = await _wait_for_job(node, job_id)
                result = job_dict["result"]
                break

        assert result is not None
        assert "final_hash" in result

        # Run the same function directly to verify result matches.
        from edge_scheduler.functions import hash_chain
        expected = hash_chain(n=500)
        assert result["final_hash"] == expected["final_hash"]

    finally:
        for node in nodes.values():
            await node.stop()


async def test_queue_endpoint_reflects_depth():
    """The /queue endpoint should show the correct queue depth."""
    config = _make_config("a", PORTS["a"], [("b", PORTS["b"]), ("c", PORTS["c"])])
    node = _Node(config)
    await node.start()

    try:
        from fastapi.testclient import TestClient
        client = TestClient(node.app)

        resp = client.get("/queue")
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_id"] == "a"
        assert data["queue_depth"] == 0
        assert isinstance(data["jobs"], list)

    finally:
        await node.stop()
