"""Scheduling logic: pick the cheapest node using the active ScoringConfig.

M5 changes:
  - pick_node now returns the full candidate list so callers can record stats.
  - SchedulerStats and DecisionLog are updated on every decision.
"""

from __future__ import annotations

import random

import structlog

from edge_scheduler.models import NodeMetrics
from edge_scheduler.peers import PeerTable
from edge_scheduler.scoring import ConfigStore, ScoringConfig, cost
from edge_scheduler.stats import DecisionLog, SchedulerStats

log = structlog.get_logger()


def pick_node(
    self_node_id: str,
    self_queue_depth: int,
    self_metrics: NodeMetrics | None,
    table: PeerTable,
    config_store: ConfigStore,
    stats: SchedulerStats | None = None,
    decision_log: DecisionLog | None = None,
) -> tuple[str, bool]:
    """Choose the cheapest node to run a function on.

    Returns:
        (node_id, forwarded) — forwarded=True means we chose a peer.
    """
    cfg = config_store.current

    self_cpu    = self_metrics.cpu_percent  if self_metrics else 0.0
    self_energy = self_metrics.energy_proxy if self_metrics else 0.0

    candidates: list[tuple[str, float]] = [
        (self_node_id, cost(self_queue_depth, 0.0, self_cpu, self_energy, cfg))
    ]

    for peer in table.all().values():
        if peer.status() != "alive":
            continue
        window = table.get_rtt_window(peer.node_id)
        net_quality = (
            window.quality_score(cfg.variance_penalty)
            if window and window._samples
            else (peer.rtt_ms or 0.0)
        )
        peer_cpu    = peer.metrics.cpu_percent  if peer.metrics else 0.0
        peer_energy = peer.metrics.energy_proxy if peer.metrics else 0.0
        peer_cost   = cost(peer.queue_depth, net_quality, peer_cpu, peer_energy, cfg)
        candidates.append((peer.node_id, peer_cost))

    min_cost = min(c for _, c in candidates)
    tied     = [n for n, c in candidates if c <= min_cost + cfg.jitter_threshold]
    chosen   = random.choice(tied)
    forwarded = chosen != self_node_id

    log.debug(
        "scheduling_decision",
        candidates=[(n, round(c, 2)) for n, c in candidates],
        tied=tied,
        chosen=chosen,
        policy=cfg.policy,
        node=self_node_id,
    )

    # M5: record stats and decision log.
    if stats is not None:
        stats.record(chosen, self_node_id, min_cost)
    if decision_log is not None:
        decision_log.record(self_node_id, chosen, forwarded, candidates, cfg.policy)

    return chosen, forwarded
