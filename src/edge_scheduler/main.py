"""Entry point: python -m edge_scheduler.main <path-to-config.yaml>"""

from __future__ import annotations

import asyncio
import sys

import structlog
import uvicorn

from edge_scheduler.api import create_app
from edge_scheduler.config import load_config
from edge_scheduler.gossip import gossip_loop
from edge_scheduler.jobs import JobStore
from edge_scheduler.metrics import MetricsSampler
from edge_scheduler.peers import PeerTable
from edge_scheduler.scoring import ConfigStore
from edge_scheduler.stats import DecisionLog, SchedulerStats

log = structlog.get_logger()


async def run(config_path: str) -> None:
    config = load_config(config_path)
    log.info("node_starting", node_id=config.node_id, host=config.host, port=config.port)

    table        = PeerTable(config)
    jobs         = JobStore()
    sampler      = MetricsSampler(config.node_id)
    config_store = ConfigStore(config.scoring)   # M4: seed from YAML
    stats        = SchedulerStats()              # M5
    decision_log = DecisionLog()                 # M5
    app          = create_app(
        table, jobs, sampler, config_store, config.node_id,
        stats=stats, decision_log=decision_log,  # M5
    )

    uvi_config = uvicorn.Config(
        app, host=config.host, port=config.port, log_level="info"
    )
    server = uvicorn.Server(uvi_config)

    gossip_task  = asyncio.create_task(
        gossip_loop(table, config.node_id, jobs, sampler, config_store)
    )
    metrics_task = asyncio.create_task(sampler.run())

    try:
        await server.serve()
    finally:
        for task in (gossip_task, metrics_task):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m edge_scheduler.main <config.yaml>", file=sys.stderr)
        sys.exit(1)
    asyncio.run(run(sys.argv[1]))


if __name__ == "__main__":
    main()
