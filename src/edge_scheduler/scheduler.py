"""Scheduling logic: pick the cheapest node using the active ScoringConfig.

M5 changes:
  - pick_node now returns the full candidate list so callers can record stats.
  - SchedulerStats and DecisionLog are updated on every decision.

M6 changes:
  - self_priority parameter: this node's configured priority (1-10).
  - Battery awareness: battery nodes penalised when below threshold.
  - Priority offset: higher-priority nodes get a cost reduction.
  - Congestion detection: sudden RTT spikes amplified via congestion_score().
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
    self_priority: int = 5,                  # M6: this node's configured priority
) -> tuple[str, bool]:
    """Choose the cheapest node to run a function on.

    Returns:
        (node_id, forwarded) — forwarded=True means we chose a peer.
    """
    cfg = config_store.current

    self_cpu     = self_metrics.cpu_percent     if self_metrics else 0.0
    self_energy  = self_metrics.energy_proxy    if self_metrics else 0.0
    self_power   = self_metrics.power_source    if self_metrics else "ac"
    self_battery = self_metrics.battery_percent if self_metrics else None

    candidates: list[tuple[str, float]] = [
        (self_node_id, cost(
            self_queue_depth, 0.0, self_cpu, self_energy, cfg,
            power_source=self_power,
            battery_percent=self_battery,
            priority=self_priority,
        ))
    ]

    for peer in table.all().values():
        if peer.status() != "alive":
            continue
        window = table.get_rtt_window(peer.node_id)
        # M6: congestion_score amplifies quality_score on sudden RTT spikes.
        net_quality = (
            window.congestion_score(cfg.variance_penalty, cfg.congestion_multiplier)
            if window and window._samples
            else (peer.rtt_ms or 0.0)
        )
        peer_cpu     = peer.metrics.cpu_percent     if peer.metrics else 0.0
        peer_energy  = peer.metrics.energy_proxy    if peer.metrics else 0.0
        peer_power   = peer.metrics.power_source    if peer.metrics else "ac"
        peer_battery = peer.metrics.battery_percent if peer.metrics else None
        peer_cost    = cost(
            peer.queue_depth, net_quality, peer_cpu, peer_energy, cfg,
            power_source=peer_power,
            battery_percent=peer_battery,
            priority=peer.priority,
        )
        candidates.append((peer.node_id, peer_cost))

    min_cost  = min(c for _, c in candidates)
    tied      = [n for n, c in candidates if c <= min_cost + cfg.jitter_threshold]
    chosen    = random.choice(tied)
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
