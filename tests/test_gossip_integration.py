"""Integration test: three in-process nodes gossip, then one is killed.

This test takes ~15 seconds — it waits for real gossip rounds and timeouts.
"""

from __future__ import annotations

import asyncio

import httpx
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
# Helpers
# ---------------------------------------------------------------------------

def _make_config(node_id: str, port: int, peers: list[tuple[str, int]]) -> NodeConfig:
    return NodeConfig(
        node_id=node_id,
        host="127.0.0.1",
        port=port,
        peers=[PeerEntry(node_id=pid, host="127.0.0.1", port=pp) for pid, pp in peers],
    )


class _Node:
    """Wraps a FastAPI server + gossip task so we can start/stop them."""

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
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="warning",
        )
        self._server = uvicorn.Server(uvi_config)
        self._serve_task   = asyncio.create_task(self._server.serve())
        self._gossip_task  = asyncio.create_task(
            gossip_loop(self.table, self.config.node_id, self.jobs,
                        self.sampler, self.config_store)
        )
        self._metrics_task = asyncio.create_task(self.sampler.run())
        # Wait until the server is accepting connections.
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


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

PORTS = {"a": 18001, "b": 18002, "c": 18003}


@pytest.mark.timeout(30)
async def test_gossip_liveness_cycle():
    """Spin up 3 nodes, verify alive, kill one, verify stale then dead."""

    configs = {
        "a": _make_config("a", PORTS["a"], [("b", PORTS["b"]), ("c", PORTS["c"])]),
        "b": _make_config("b", PORTS["b"], [("a", PORTS["a"]), ("c", PORTS["c"])]),
        "c": _make_config("c", PORTS["c"], [("a", PORTS["a"]), ("b", PORTS["b"])]),
    }

    nodes = {nid: _Node(cfg) for nid, cfg in configs.items()}

    # Start all three nodes.
    for node in nodes.values():
        await node.start()

    try:
        # Let gossip settle (~3 seconds = at least one full round).
        await asyncio.sleep(3)

        # -- Assert: node A sees B and C as alive with rtt_ms --
        snap_a = nodes["a"].table.snapshot()
        by_id = {p["node_id"]: p for p in snap_a}
        assert by_id["b"]["status"] == "alive", f"expected b alive, got {by_id['b']}"
        assert by_id["c"]["status"] == "alive", f"expected c alive, got {by_id['c']}"
        assert by_id["b"]["rtt_ms"] is not None
        assert by_id["c"]["rtt_ms"] is not None

        # -- Kill node B --
        await nodes["b"].stop()

        # Wait 7 seconds — B should be stale (last_seen > 6s ago).
        await asyncio.sleep(7)
        snap_a = nodes["a"].table.snapshot()
        by_id = {p["node_id"]: p for p in snap_a}
        assert by_id["b"]["status"] == "stale", f"expected b stale, got {by_id['b']}"
        assert by_id["c"]["status"] == "alive", f"expected c alive, got {by_id['c']}"

        # Wait 5 more seconds — B should be dead (last_seen > 10s ago).
        await asyncio.sleep(5)
        snap_a = nodes["a"].table.snapshot()
        by_id = {p["node_id"]: p for p in snap_a}
        assert by_id["b"]["status"] == "dead", f"expected b dead, got {by_id['b']}"
        assert by_id["c"]["status"] == "alive", f"expected c alive, got {by_id['c']}"

    finally:
        # Clean shutdown of remaining nodes.
        for node in nodes.values():
            await node.stop()
