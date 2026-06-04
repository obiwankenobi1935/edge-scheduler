"""Adaptive policy controller: automatically switches scheduling mode.

Runs as a background task alongside gossip and metrics sampling.
Every 2 seconds it evaluates the node's current conditions and switches
the active ScoringConfig policy if a threshold is crossed.

Priority order (highest wins):
  1. local_first  — battery critical (<10%), no alive peers, or all peers congested
  2. energy_first — battery low (<30%)
  3. latency_first — CPU sustained high (>80%)
  4. balanced      — default

Hysteresis: a condition must be detected on HYSTERESIS_COUNT consecutive
ticks before the policy actually switches. This prevents rapid flapping
when a metric sits on a threshold boundary.
"""

from __future__ import annotations

import asyncio

import structlog

from edge_scheduler.metrics import MetricsSampler
from edge_scheduler.peers import PeerTable
from edge_scheduler.scoring import ConfigStore, Policy

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

BATTERY_CRITICAL  = 10.0   # % — below this → local_first
BATTERY_LOW       = 30.0   # % — below this (on battery) → energy_first
CPU_HIGH          = 80.0   # % — above this → latency_first
HYSTERESIS_COUNT  = 2      # consecutive ticks required before switching
CHECK_INTERVAL_S  = 2.0    # same cadence as gossip and metrics


class AdaptiveController:
    """Watches local conditions and switches policy presets automatically."""

    def __init__(
        self,
        config_store: ConfigStore,
        sampler: MetricsSampler,
        table: PeerTable,
        node_id: str,
    ) -> None:
        self.config_store = config_store
        self.sampler      = sampler
        self.table        = table
        self.node_id      = node_id

        # Hysteresis state
        self._candidate: Policy = "balanced"
        self._candidate_count: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _all_peers_congested(self, alive_peers: list) -> bool:
        """True only when every alive peer has a congested RTT window."""
        if not alive_peers:
            return False
        for peer in alive_peers:
            window = self.table.get_rtt_window(peer.node_id)
            if not (window and window.is_congested):
                return False
        return True

    def _detect_target(self) -> Policy:
        """Evaluate current conditions and return the desired policy."""
        m     = self.sampler.latest
        peers = list(self.table.all().values())
        alive = [p for p in peers if p.status() == "alive"]

        on_battery      = m.power_source == "battery"
        battery_known   = m.battery_percent is not None
        battery_pct     = m.battery_percent or 100.0   # default safe if unknown

        # ── Priority 1: emergency → local_first ──────────────────────────
        if on_battery and battery_known and battery_pct < BATTERY_CRITICAL:
            return "local_first"

        if peers and not alive:          # all known peers are dead
            return "local_first"

        if self._all_peers_congested(alive):
            return "local_first"

        # ── Priority 2: save power → energy_first ────────────────────────
        if on_battery and battery_known and battery_pct < BATTERY_LOW:
            return "energy_first"

        # ── Priority 3: high CPU → latency_first ─────────────────────────
        # When this node is under heavy CPU load, forward aggressively to
        # whichever peer responds fastest (minimise network penalty).
        if m.cpu_percent > CPU_HIGH:
            return "latency_first"

        # ── Priority 4: default ───────────────────────────────────────────
        return "balanced"

    def _detect_reason(self) -> str:
        """Human-readable explanation of the current target policy."""
        m     = self.sampler.latest
        peers = list(self.table.all().values())
        alive = [p for p in peers if p.status() == "alive"]

        on_battery    = m.power_source == "battery"
        battery_known = m.battery_percent is not None
        battery_pct   = m.battery_percent or 100.0

        if on_battery and battery_known and battery_pct < BATTERY_CRITICAL:
            return f"battery critical ({battery_pct:.0f}%)"
        if peers and not alive:
            return "no alive peers"
        if self._all_peers_congested(alive):
            return "all peers congested"
        if on_battery and battery_known and battery_pct < BATTERY_LOW:
            return f"battery low ({battery_pct:.0f}%)"
        if m.cpu_percent > CPU_HIGH:
            return f"cpu high ({m.cpu_percent:.0f}%)"
        return "normal conditions"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self) -> Policy | None:
        """Evaluate once. Returns the new policy name if a switch happened."""
        target  = self._detect_target()
        current = self.config_store.current.policy

        if target == self._candidate:
            self._candidate_count += 1
        else:
            # Condition changed — reset hysteresis counter
            self._candidate       = target
            self._candidate_count = 1

        if self._candidate_count >= HYSTERESIS_COUNT and target != current:
            self.config_store.set_policy(target)
            log.info(
                "adaptive_policy_switch",
                node=self.node_id,
                from_policy=current,
                to_policy=target,
                reason=self._detect_reason(),
                battery_pct=self.sampler.latest.battery_percent,
                cpu_pct=self.sampler.latest.cpu_percent,
            )
            return target

        return None

    async def run(self) -> None:
        """Background loop — launch with asyncio.create_task()."""
        log.info("adaptive_controller_started", node=self.node_id)
        while True:
            self.step()
            await asyncio.sleep(CHECK_INTERVAL_S)
