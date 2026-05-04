"""Unit tests for PeerRecord status computation."""

from datetime import datetime, timedelta, timezone

from edge_scheduler.models import PeerRecord, peer_status


def _make_peer(last_seen: datetime) -> PeerRecord:
    return PeerRecord(node_id="x", host="127.0.0.1", port=9999, last_seen=last_seen)


class TestPeerStatus:
    """Status is computed from the gap between last_seen and now."""

    def test_alive_at_0s(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        peer = _make_peer(last_seen=now)
        assert peer.status(now=now) == "alive"

    def test_alive_at_5s(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        peer = _make_peer(last_seen=now - timedelta(seconds=5))
        assert peer.status(now=now) == "alive"

    def test_alive_at_boundary_6s(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        peer = _make_peer(last_seen=now - timedelta(seconds=6))
        assert peer.status(now=now) == "alive"

    def test_stale_at_7s(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        peer = _make_peer(last_seen=now - timedelta(seconds=7))
        assert peer.status(now=now) == "stale"

    def test_stale_at_boundary_10s(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        peer = _make_peer(last_seen=now - timedelta(seconds=10))
        assert peer.status(now=now) == "stale"

    def test_dead_at_11s(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        peer = _make_peer(last_seen=now - timedelta(seconds=11))
        assert peer.status(now=now) == "dead"

    def test_dead_at_60s(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        peer = _make_peer(last_seen=now - timedelta(seconds=60))
        assert peer.status(now=now) == "dead"


class TestPeerStatusFunction:
    """Same logic via the standalone peer_status() helper."""

    def test_alive(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert peer_status(now - timedelta(seconds=3), now=now) == "alive"

    def test_stale(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert peer_status(now - timedelta(seconds=8), now=now) == "stale"

    def test_dead(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert peer_status(now - timedelta(seconds=15), now=now) == "dead"
