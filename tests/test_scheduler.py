"""Unit tests for the cost function and pick_node scheduling logic."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from edge_scheduler.models import NodeMetrics, PeerRecord
from edge_scheduler.scheduler import pick_node
from edge_scheduler.scoring import (
    ConfigStore,
    ScoringConfig,
    cost,
)

# Expose constants from ScoringConfig defaults for test assertions
_default = ScoringConfig()
QUEUE_WEIGHT     = _default.queue_weight
RTT_WEIGHT       = _default.network_weight
CPU_WEIGHT       = _default.cpu_weight
ENERGY_WEIGHT    = _default.energy_weight
JITTER_THRESHOLD = _default.jitter_threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metrics(cpu: float = 0.0, energy: float = 0.0) -> NodeMetrics:
    return NodeMetrics(
        node_id="x",
        cpu_percent=cpu,
        memory_percent=50.0,
        memory_available_mb=4096.0,
        energy_proxy=energy,
        sampled_at=datetime.now(timezone.utc),
    )


def _alive_peer(
    node_id: str,
    queue_depth: int = 0,
    rtt_ms: float = 1.0,
    cpu: float = 0.0,
    energy: float = 0.0,
) -> PeerRecord:
    return PeerRecord(
        node_id=node_id,
        host="127.0.0.1",
        port=9000,
        last_seen=datetime.now(timezone.utc),
        rtt_ms=rtt_ms,
        queue_depth=queue_depth,
        metrics=_make_metrics(cpu=cpu, energy=energy),
    )


def _dead_peer(node_id: str) -> PeerRecord:
    from datetime import timedelta
    return PeerRecord(
        node_id=node_id,
        host="127.0.0.1",
        port=9000,
        last_seen=datetime.now(timezone.utc) - timedelta(seconds=60),
        rtt_ms=5.0,
        queue_depth=0,
        metrics=None,
    )


def _mock_table(peers: list[PeerRecord]) -> MagicMock:
    table = MagicMock()
    table.all.return_value = {p.node_id: p for p in peers}
    # get_rtt_window returns None by default so the scheduler falls back to raw RTT.
    table.get_rtt_window.return_value = None
    return table


# ---------------------------------------------------------------------------
# cost()
# ---------------------------------------------------------------------------

class TestCostFunction:
    def test_zero_everything(self):
        assert cost(0, 0.0, 0.0, 0.0) == 0.0

    def test_queue_only(self):
        assert cost(3, 0.0) == pytest.approx(3 * QUEUE_WEIGHT)

    def test_rtt_only(self):
        assert cost(0, 20.0) == pytest.approx(20.0 * RTT_WEIGHT)

    def test_cpu_only(self):
        assert cost(0, 0.0, cpu_percent=50.0) == pytest.approx(50.0 * CPU_WEIGHT)

    def test_energy_only(self):
        assert cost(0, 0.0, energy_proxy=40.0) == pytest.approx(40.0 * ENERGY_WEIGHT)

    def test_all_combined(self):
        expected = (
            (2 * QUEUE_WEIGHT)
            + (10.0 * RTT_WEIGHT)
            + (60.0 * CPU_WEIGHT)
            + (30.0 * ENERGY_WEIGHT)
        )
        assert cost(2, 10.0, 60.0, 30.0) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# pick_node()
# ---------------------------------------------------------------------------

class TestPickNode:
    def _store(self) -> ConfigStore:
        return ConfigStore()

    def test_picks_self_when_no_alive_peers(self):
        table = _mock_table([_dead_peer("b")])
        node, forwarded = pick_node("a", 0, _make_metrics(), table, self._store())
        assert node == "a"
        assert forwarded is False

    def test_picks_self_when_clearly_cheapest(self):
        table = _mock_table([_alive_peer("b", queue_depth=5, rtt_ms=2.0)])
        node, forwarded = pick_node("a", 0, _make_metrics(), table, self._store())
        assert node == "a"
        assert forwarded is False

    def test_picks_peer_when_clearly_cheapest(self):
        table = _mock_table([_alive_peer("b", queue_depth=0, rtt_ms=2.0)])
        node, forwarded = pick_node("a", 10, _make_metrics(), table, self._store())
        assert node == "b"
        assert forwarded is True

    def test_near_tie_randomises(self):
        table = _mock_table([_alive_peer("b", queue_depth=0, rtt_ms=2.0)])
        results = {pick_node("a", 0, _make_metrics(), table, self._store())[0] for _ in range(50)}
        assert "a" in results
        assert "b" in results

    def test_high_cpu_peer_penalised(self):
        table = _mock_table([_alive_peer("b", queue_depth=0, rtt_ms=1.0, cpu=80.0)])
        node, forwarded = pick_node("a", 0, _make_metrics(cpu=0.0), table, self._store())
        assert node == "a"
        assert forwarded is False

    def test_no_metrics_peer_defaults_to_zero(self):
        peer = _alive_peer("b", queue_depth=0, rtt_ms=2.0)
        peer.metrics = None
        table = _mock_table([peer])
        node, forwarded = pick_node("a", 0, _make_metrics(), table, self._store())
        assert node in ("a", "b")

    def test_dead_peers_never_chosen(self):
        table = _mock_table([_dead_peer("b"), _alive_peer("c", rtt_ms=2.0)])
        results = {pick_node("a", 0, _make_metrics(), table, self._store())[0] for _ in range(30)}
        assert "b" not in results

    def test_forwarded_flag_consistent(self):
        table = _mock_table([_alive_peer("b", rtt_ms=2.0)])
        for _ in range(30):
            node, forwarded = pick_node("a", 0, _make_metrics(), table, self._store())
            assert forwarded == (node != "a")

    def test_custom_weights_affect_decision(self):
        # With a very high cpu_weight, a high-CPU peer should always lose.
        cfg = ScoringConfig(cpu_weight=100.0, jitter_threshold=5.0)
        store = ConfigStore(cfg)
        # Peer b: cpu=50 → cpu cost = 50*100 = 5000. Self cpu=0 → cost=0. Self always wins.
        table = _mock_table([_alive_peer("b", queue_depth=0, rtt_ms=1.0, cpu=50.0)])
        node, _ = pick_node("a", 0, _make_metrics(cpu=0.0), table, store)
        assert node == "a"
