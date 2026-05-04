"""Pydantic models for peer discovery, gossip, and job invocation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel

from edge_scheduler.scoring import ScoringConfig


# ---------------------------------------------------------------------------
# M3: local resource metrics
# ---------------------------------------------------------------------------

class NodeMetrics(BaseModel):
    node_id: str
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    energy_proxy: float           # cpu * memory / 100 on laptop; real watts on Pi
    sampled_at: datetime

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_available_mb": self.memory_available_mb,
            "energy_proxy": self.energy_proxy,
            "sampled_at": self.sampled_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Gossip message: sent as the body of POST /peer/update
# ---------------------------------------------------------------------------

class PeerUpdate(BaseModel):
    from_node_id: str
    sent_at: datetime
    queue_depth: int = 0                          # M2: sender broadcasts current load
    metrics: NodeMetrics | None = None            # M3: sender broadcasts resource state
    scoring_config: ScoringConfig | None = None   # M4: sender broadcasts active weights


# ---------------------------------------------------------------------------
# Internal record kept per known peer
# ---------------------------------------------------------------------------

ALIVE_THRESHOLD_S = 6.0   # <= 6 s since last_seen  -> alive
STALE_THRESHOLD_S = 10.0  # <= 10 s since last_seen -> stale, else dead


def peer_status(last_seen: datetime, now: datetime | None = None) -> Literal["alive", "stale", "dead"]:
    """Compute a peer's liveness status from its last_seen timestamp.

    *now* can be injected for deterministic testing; defaults to UTC now.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    delta = (now - last_seen).total_seconds()
    if delta <= ALIVE_THRESHOLD_S:
        return "alive"
    if delta <= STALE_THRESHOLD_S:
        return "stale"
    return "dead"


class PeerRecord(BaseModel):
    node_id: str
    host: str
    port: int
    last_seen: datetime
    rtt_ms: float | None = None
    queue_depth: int = 0                  # M2: last known queue depth from gossip
    metrics: NodeMetrics | None = None    # M3: last known resource state from gossip

    def status(self, now: datetime | None = None) -> Literal["alive", "stale", "dead"]:
        return peer_status(self.last_seen, now)

    def to_dict(self, now: datetime | None = None) -> dict:
        """Serialize for the /peers JSON response, including computed status."""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "last_seen": self.last_seen.isoformat(),
            "rtt_ms": self.rtt_ms,
            "queue_depth": self.queue_depth,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "status": self.status(now),
        }


# ---------------------------------------------------------------------------
# Invocation models: /invoke and /execute endpoints
# ---------------------------------------------------------------------------

class InvokeRequest(BaseModel):
    function: str           # "matrix_multiply" | "prime_search" | "hash_chain"
    args: dict[str, Any]


class InvokeResponse(BaseModel):
    job_id: str
    scheduled_on: str       # node_id that will run the function
    forwarded: bool         # True if this node forwarded to a peer
    status: Literal["accepted"] = "accepted"


class ExecuteRequest(BaseModel):
    job_id: str
    function: str
    args: dict[str, Any]
    origin_node: str        # node_id that originally received /invoke


# ---------------------------------------------------------------------------
# Job record
# ---------------------------------------------------------------------------

class JobRecord(BaseModel):
    job_id: str
    function: str
    args: dict[str, Any]
    status: Literal["pending", "running", "done", "failed"] = "pending"
    result: Any | None = None
    error: str | None = None
    scheduled_on: str
    origin_node: str
    created_at: datetime
    finished_at: datetime | None = None
    duration_ms: float | None = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "function": self.function,
            "args": self.args,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "scheduled_on": self.scheduled_on,
            "origin_node": self.origin_node,
            "created_at": self.created_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_ms": self.duration_ms,
        }
