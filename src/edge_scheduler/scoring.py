"""M4/M5: Configurable multi-metric scoring.

Key additions over M3:
  - ScoringConfig  : all weights + policy in one place, with defaults
  - RTTWindow      : rolling window of last N RTT samples per peer
  - network_quality: mean + (stddev × variance_penalty) — punishes instability
  - ConfigStore    : holds the live config; adopts a gossiped config only if
                     it is strictly newer (last-write-wins by timestamp)

M5 additions:
  - policy field   : named preset that auto-applies weight profiles
  - POLICY_PRESETS : balanced / latency_first / energy_first / local_first
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timezone
from typing import Literal

import structlog
from pydantic import BaseModel, Field

log = structlog.get_logger()

RTT_WINDOW_SIZE = 10

Policy = Literal["balanced", "latency_first", "energy_first", "local_first"]

# ---------------------------------------------------------------------------
# Policy presets — each is a dict of weight overrides applied on top of
# the defaults when a policy is selected.
# ---------------------------------------------------------------------------

POLICY_PRESETS: dict[str, dict] = {
    "balanced": {
        "queue_weight":   10.0,
        "network_weight":  1.0,
        "cpu_weight":      0.5,
        "energy_weight":   0.3,
    },
    "latency_first": {
        "queue_weight":    5.0,
        "network_weight": 10.0,   # network dominates
        "cpu_weight":      0.1,
        "energy_weight":   0.1,
    },
    "energy_first": {
        "queue_weight":    5.0,
        "network_weight":  0.5,
        "cpu_weight":      0.5,
        "energy_weight":  10.0,   # energy dominates
    },
    "local_first": {
        "queue_weight":  100.0,   # massive queue penalty → almost never forward
        "network_weight": 50.0,   # high network cost → further discourages forwarding
        "cpu_weight":      0.1,
        "energy_weight":   0.1,
    },
}


# ---------------------------------------------------------------------------
# ScoringConfig
# ---------------------------------------------------------------------------

class ScoringConfig(BaseModel):
    policy:           Policy  = "balanced"
    queue_weight:     float   = 10.0
    network_weight:   float   =  1.0
    cpu_weight:       float   =  0.5
    energy_weight:    float   =  0.3
    variance_penalty: float   =  2.0
    jitter_threshold: float   =  5.0
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict:
        return {
            "policy":           self.policy,
            "queue_weight":     self.queue_weight,
            "network_weight":   self.network_weight,
            "cpu_weight":       self.cpu_weight,
            "energy_weight":    self.energy_weight,
            "variance_penalty": self.variance_penalty,
            "jitter_threshold": self.jitter_threshold,
            "updated_at":       self.updated_at.isoformat(),
        }


def apply_policy(policy: Policy) -> ScoringConfig:
    """Return a ScoringConfig with preset weights for the given policy."""
    preset = POLICY_PRESETS.get(policy, POLICY_PRESETS["balanced"])
    return ScoringConfig(policy=policy, **preset)


# ---------------------------------------------------------------------------
# RTTWindow
# ---------------------------------------------------------------------------

class RTTWindow:
    """Keeps the last RTT_WINDOW_SIZE RTT measurements for one peer."""

    def __init__(self) -> None:
        self._samples: deque[float] = deque(maxlen=RTT_WINDOW_SIZE)

    def add(self, rtt_ms: float) -> None:
        self._samples.append(rtt_ms)

    def mean(self) -> float:
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    def stddev(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        m = self.mean()
        variance = sum((x - m) ** 2 for x in self._samples) / len(self._samples)
        return math.sqrt(variance)

    def quality_score(self, variance_penalty: float) -> float:
        return self.mean() + (self.stddev() * variance_penalty)

    def snapshot(self) -> dict:
        return {
            "mean_ms":   round(self.mean(), 3),
            "stddev_ms": round(self.stddev(), 3),
            "samples":   len(self._samples),
        }


# ---------------------------------------------------------------------------
# Cost function
# ---------------------------------------------------------------------------

def cost(
    queue_depth:     int,
    network_quality: float,
    cpu_percent:     float = 0.0,
    energy_proxy:    float = 0.0,
    config:          ScoringConfig | None = None,
) -> float:
    cfg = config or ScoringConfig()
    return (
          (queue_depth     * cfg.queue_weight)
        + (network_quality * cfg.network_weight)
        + (cpu_percent     * cfg.cpu_weight)
        + (energy_proxy    * cfg.energy_weight)
    )


# ---------------------------------------------------------------------------
# ConfigStore
# ---------------------------------------------------------------------------

class ConfigStore:
    """Holds the active ScoringConfig. Thread-safe for single-threaded asyncio."""

    def __init__(self, initial: ScoringConfig | None = None) -> None:
        self.current: ScoringConfig = initial or ScoringConfig()

    def update_local(self, new: ScoringConfig) -> None:
        """Called by POST /scheduler/config. Always accepts the new config."""
        new_with_ts = new.model_copy(
            update={"updated_at": datetime.now(timezone.utc)}
        )
        self.current = new_with_ts
        log.info("scoring_config_updated_local", policy=new_with_ts.policy,
                 weights=new_with_ts.to_dict())

    def set_policy(self, policy: Policy) -> None:
        """Convenience: switch policy and apply its preset weights."""
        new_cfg = apply_policy(policy)
        self.update_local(new_cfg)
        log.info("policy_switched", policy=policy)

    def merge_gossip(self, incoming: ScoringConfig) -> bool:
        """Accepts a gossiped config only if it is strictly newer."""
        if incoming.updated_at > self.current.updated_at:
            self.current = incoming
            log.info("scoring_config_adopted_from_gossip",
                     policy=incoming.policy,
                     updated_at=incoming.updated_at.isoformat())
            return True
        return False
