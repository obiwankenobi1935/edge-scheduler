"""Local metrics sampler: reads psutil every 2 seconds.

M3 collects four metrics per node:
  - cpu_percent          : 0-100, from psutil
  - memory_percent       : 0-100, from psutil
  - memory_available_mb  : available RAM in MB
  - energy_proxy         : cpu * memory / 100 (laptop approximation)

In M6/M7 on real Raspberry Pis, energy_proxy gets replaced with actual
watt readings from an INA219 sensor or the Pi's onboard PMIC — the rest
of this module and all callers stay unchanged.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import psutil
import structlog

from edge_scheduler.models import NodeMetrics

log = structlog.get_logger()

SAMPLE_INTERVAL_S = 2.0   # same cadence as gossip


def _sample_now(node_id: str) -> NodeMetrics:
    """Take a single psutil reading and return a NodeMetrics instance."""
    cpu = psutil.cpu_percent(interval=None)   # non-blocking; uses last interval
    vm  = psutil.virtual_memory()
    mem_pct = vm.percent
    mem_avail_mb = vm.available / (1024 * 1024)
    energy = cpu * mem_pct / 100.0            # synthetic proxy

    return NodeMetrics(
        node_id=node_id,
        cpu_percent=round(cpu, 2),
        memory_percent=round(mem_pct, 2),
        memory_available_mb=round(mem_avail_mb, 1),
        energy_proxy=round(energy, 2),
        sampled_at=datetime.now(timezone.utc),
    )


class MetricsSampler:
    """Holds the latest local metrics sample. Thread-safe for reads.

    Background loop calls _sample_now() every SAMPLE_INTERVAL_S and stores
    the result. Gossip loop and /metrics endpoint both read from .latest.
    """

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        # Take one immediate sample so .latest is never None.
        self.latest: NodeMetrics = _sample_now(node_id)

    async def run(self) -> None:
        """Background sampling loop — launch with asyncio.create_task()."""
        # Prime psutil's CPU measurement. cpu_percent(interval=None) returns 0.0
        # on the very first call because it has no previous interval to compare
        # against. Calling it once here and discarding the result means every
        # subsequent call returns a real value.
        psutil.cpu_percent(interval=None)
        await asyncio.sleep(SAMPLE_INTERVAL_S)

        while True:
            self.latest = _sample_now(self.node_id)
            log.debug(
                "metrics_sampled",
                cpu=self.latest.cpu_percent,
                mem=self.latest.memory_percent,
                energy=self.latest.energy_proxy,
            )
            await asyncio.sleep(SAMPLE_INTERVAL_S)
