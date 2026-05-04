"""Background gossip loop: POST /peer/update to every peer every 2 seconds."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
import structlog

from edge_scheduler.jobs import JobStore
from edge_scheduler.metrics import MetricsSampler
from edge_scheduler.peers import PeerTable
from edge_scheduler.scoring import ConfigStore

log = structlog.get_logger()

GOSSIP_INTERVAL_S = 2.0
REQUEST_TIMEOUT_S = 1.0


async def _gossip_one(
    client: httpx.AsyncClient,
    peer_id: str,
    url: str,
    payload: dict,
    table: PeerTable,
) -> None:
    """Send a single gossip POST and record the result."""
    try:
        start = asyncio.get_event_loop().time()
        resp = await client.post(url, json=payload, timeout=REQUEST_TIMEOUT_S)
        rtt_ms = (asyncio.get_event_loop().time() - start) * 1000
        resp.raise_for_status()
        table.update_last_seen(peer_id, rtt_ms=rtt_ms)
        log.debug("gossip_ok", peer=peer_id, rtt_ms=round(rtt_ms, 2))
    except Exception as exc:
        log.info("gossip_fail", peer=peer_id, error=type(exc).__name__)


async def gossip_loop(
    table: PeerTable,
    node_id: str,
    jobs: JobStore,
    sampler: MetricsSampler,
    config_store: ConfigStore,      # M4: broadcast active weights
) -> None:
    """Run forever, gossiping to all peers every GOSSIP_INTERVAL_S seconds."""
    async with httpx.AsyncClient() as client:
        while True:
            peers = table.all()
            payload = {
                "from_node_id": node_id,
                "sent_at": datetime.now(timezone.utc).isoformat(),
                "queue_depth": jobs.queue_depth,
                "metrics": sampler.latest.model_dump(mode="json"),
                "scoring_config": config_store.current.model_dump(mode="json"),  # M4
            }
            tasks = [
                _gossip_one(
                    client,
                    pid,
                    f"http://{p.host}:{p.port}/peer/update",
                    payload,
                    table,
                )
                for pid, p in peers.items()
            ]
            await asyncio.gather(*tasks)
            await asyncio.sleep(GOSSIP_INTERVAL_S)
