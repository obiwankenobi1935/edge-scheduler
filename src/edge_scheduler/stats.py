"""M5: Scheduler statistics and decision log.

SchedulerStats  — running counters for thesis evaluation data.
DecisionLog     — rolling window of the last N scheduling decisions.

Both are in-memory per node. No persistence across restarts.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any

DECISION_LOG_SIZE = 50   # keep last 50 decisions per node


class SchedulerStats:
    """Running counters updated on every /invoke call."""

    def __init__(self) -> None:
        self.total_invocations: int = 0
        self.local_executions:  int = 0
        self.forwarded:         int = 0
        self.forwarded_to:      dict[str, int] = {}
        self._cost_scores:      list[float] = []

    def record(
        self,
        chosen_node: str,
        self_node_id: str,
        cost_score: float,
    ) -> None:
        self.total_invocations += 1
        self._cost_scores.append(cost_score)
        if chosen_node == self_node_id:
            self.local_executions += 1
        else:
            self.forwarded += 1
            self.forwarded_to[chosen_node] = (
                self.forwarded_to.get(chosen_node, 0) + 1
            )

    @property
    def avg_cost_score(self) -> float:
        if not self._cost_scores:
            return 0.0
        return round(sum(self._cost_scores) / len(self._cost_scores), 3)

    def to_dict(self) -> dict:
        return {
            "total_invocations": self.total_invocations,
            "local_executions":  self.local_executions,
            "forwarded":         self.forwarded,
            "forwarded_to":      self.forwarded_to,
            "avg_cost_score":    self.avg_cost_score,
        }


class DecisionLog:
    """Rolling log of the last DECISION_LOG_SIZE scheduling decisions."""

    def __init__(self) -> None:
        self._entries: deque[dict[str, Any]] = deque(maxlen=DECISION_LOG_SIZE)

    def record(
        self,
        self_node_id: str,
        chosen_node: str,
        forwarded: bool,
        candidates: list[tuple[str, float]],
        policy: str,
    ) -> None:
        self._entries.appendleft({
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "node":         self_node_id,
            "chosen":       chosen_node,
            "forwarded":    forwarded,
            "policy":       policy,
            "candidates":   [
                {"node": n, "cost": round(c, 3)} for n, c in candidates
            ],
        })

    def to_list(self) -> list[dict]:
        return list(self._entries)
