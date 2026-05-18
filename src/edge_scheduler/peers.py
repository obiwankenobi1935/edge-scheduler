"""Peer table: holds known peers and provides lookup/update helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from edge_scheduler.config import NodeConfig
from edge_scheduler.models import NodeMetrics, PeerRecord
from edge_scheduler.scoring import RTTWindow

log = structlog.get_logger()


class PeerTable:
    """In-memory table of known peers, keyed by node_id."""

    def __init__(self, config: NodeConfig) -> None:
        self.node_id = config.node_id
        now = datetime.now(timezone.utc)
        self._peers: dict[str, PeerRecord] = {
            p.node_id: PeerRecord(
                node_id=p.node_id,
                host=p.host,
                port=p.port,
                last_seen=now,
            )
            for p in config.peers
        }
        # M4: one RTT window per peer, kept separate from PeerRecord so the
        # rolling deque doesn't need to be serialised into JSON.
        self._rtt_windows: dict[str, RTTWindow] = {
            p.node_id: RTTWindow() for p in config.peers
        }

    def get(self, node_id: str) -> PeerRecord | None:
        return self._peers.get(node_id)

    def all(self) -> dict[str, PeerRecord]:
        return dict(self._peers)

    def get_rtt_window(self, node_id: str) -> RTTWindow | None:
        return self._rtt_windows.get(node_id)

    def update_last_seen(
        self,
        node_id: str,
        rtt_ms: float | None = None,
        queue_depth: int | None = None,
        metrics: NodeMetrics | None = None,
        priority: int | None = None,           # M6: peer's configured priority
    ) -> None:
        """Mark a peer as freshly seen."""
        peer = self._peers.get(node_id)
        if peer is None:
            log.warning("peer_update_unknown", node_id=node_id)
            return
        old_status = peer.status()
        peer.last_seen = datetime.now(timezone.utc)
        if rtt_ms is not None:
            peer.rtt_ms = rtt_ms
            # M4: feed the new sample into the rolling RTT window.
            if node_id in self._rtt_windows:
                self._rtt_windows[node_id].add(rtt_ms)
        if queue_depth is not None:
            peer.queue_depth = queue_depth
        if metrics is not None:
            peer.metrics = metrics
        if priority is not None:
            peer.priority = priority             # M6: update priority from gossip
        new_status = peer.status()
        if old_status != new_status:
            log.info("peer_status_change", peer=node_id, old=old_status, new=new_status)

    def snapshot(self) -> list[dict]:
        """Return a JSON-serialisable list of all peers with computed status."""
        now = datetime.now(timezone.utc)
        result = []
        for p in self._peers.values():
            d = p.to_dict(now)
            # M4: attach RTT window stats for debugging.
            window = self._rtt_windows.get(p.node_id)
            d["rtt_window"] = window.snapshot() if window else None
            result.append(d)
        return result
