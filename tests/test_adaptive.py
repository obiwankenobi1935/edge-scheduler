"""M6 tests: battery awareness, node priority, congestion detection, and adaptive controller."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from edge_scheduler.adaptive import (
    BATTERY_CRITICAL,
    BATTERY_LOW,
    CPU_HIGH,
    HYSTERESIS_COUNT,
    AdaptiveController,
)
from edge_scheduler.models import NodeMetrics, PeerRecord
from edge_scheduler.scheduler import pick_node
from edge_scheduler.scoring import ConfigStore, RTTWindow, ScoringConfig, cost


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metrics(
    cpu: float = 0.0,
    energy: float = 0.0,
    power_source: str = "ac",
    battery_percent: float | None = None,
) -> NodeMetrics:
    return NodeMetrics(
        node_id="x",
        cpu_percent=cpu,
        memory_percent=50.0,
        memory_available_mb=4096.0,
        energy_proxy=energy,
        sampled_at=datetime.now(timezone.utc),
        power_source=power_source,
        battery_percent=battery_percent,
    )


def _alive_peer(
    node_id: str,
    queue_depth: int = 0,
    rtt_ms: float = 1.0,
    priority: int = 5,
    power_source: str = "ac",
    battery_percent: float | None = None,
) -> PeerRecord:
    return PeerRecord(
        node_id=node_id,
        host="127.0.0.1",
        port=9000,
        last_seen=datetime.now(timezone.utc),
        rtt_ms=rtt_ms,
        queue_depth=queue_depth,
        priority=priority,
        metrics=_make_metrics(power_source=power_source, battery_percent=battery_percent),
    )


from unittest.mock import MagicMock

def _mock_table(peers: list[PeerRecord], rtt_window: RTTWindow | None = None) -> MagicMock:
    table = MagicMock()
    table.all.return_value = {p.node_id: p for p in peers}
    table.get_rtt_window.return_value = rtt_window
    return table


def _default_store() -> ConfigStore:
    return ConfigStore()


# ---------------------------------------------------------------------------
# Battery awareness — cost()
# ---------------------------------------------------------------------------

class TestBatteryCost:
    def test_ac_node_no_battery_penalty(self):
        """A node on AC power with known battery should not be penalised."""
        c_ac = cost(0, 0.0, power_source="ac", battery_percent=50.0)
        c_baseline = cost(0, 0.0)
        assert c_ac == pytest.approx(c_baseline)

    def test_battery_node_penalised(self):
        """A node on battery has a higher cost than the same node on AC."""
        c_battery = cost(0, 0.0, power_source="battery", battery_percent=80.0)
        c_ac      = cost(0, 0.0, power_source="ac",      battery_percent=80.0)
        assert c_battery > c_ac

    def test_low_battery_higher_penalty_than_high_battery(self):
        """10% battery incurs a much higher cost than 80% battery."""
        c_low  = cost(0, 0.0, power_source="battery", battery_percent=10.0)
        c_high = cost(0, 0.0, power_source="battery", battery_percent=80.0)
        assert c_low > c_high

    def test_battery_none_no_penalty(self):
        """battery_percent=None on battery source → no penalty (unknown state)."""
        c_unknown = cost(0, 0.0, power_source="battery", battery_percent=None)
        c_baseline = cost(0, 0.0)
        assert c_unknown == pytest.approx(c_baseline)

    def test_battery_penalty_formula(self):
        """(100 - battery_percent) * battery_weight is the exact penalty."""
        cfg = ScoringConfig(battery_weight=3.0)
        c = cost(0, 0.0, config=cfg, power_source="battery", battery_percent=70.0)
        # Priority offset is centred at 5: (5-5)*2 = 0, so it doesn't interfere.
        assert c == pytest.approx((100.0 - 70.0) * 3.0)


# ---------------------------------------------------------------------------
# Battery awareness — pick_node()
# ---------------------------------------------------------------------------

class TestBatteryPickNode:
    def test_avoids_low_battery_peer(self):
        """Peer on 10% battery should lose to self on AC with equal queue."""
        peer = _alive_peer("b", queue_depth=0, rtt_ms=0.1,
                           power_source="battery", battery_percent=10.0)
        table = _mock_table([peer])
        node, _ = pick_node("a", 0, _make_metrics(power_source="ac"),
                            table, _default_store(), self_priority=5)
        assert node == "a"

    def test_prefers_ac_peer_over_battery_self(self):
        """If self is on low battery, prefer forwarding to an AC peer."""
        peer = _alive_peer("b", queue_depth=0, rtt_ms=1.0, power_source="ac")
        table = _mock_table([peer])
        self_metrics = _make_metrics(power_source="battery", battery_percent=10.0)
        # Self battery cost = (100-10)*3 = 270, peer cost ≈ 1*1 = 1 → peer wins
        node, forwarded = pick_node("a", 0, self_metrics, table, _default_store(),
                                    self_priority=5)
        assert node == "b"
        assert forwarded is True


# ---------------------------------------------------------------------------
# Node priority — cost()
# ---------------------------------------------------------------------------

class TestPriorityCost:
    def test_higher_priority_lower_cost(self):
        """Priority 9 node has lower cost than priority 1 node, all else equal."""
        c_high = cost(0, 0.0, priority=9)
        c_low  = cost(0, 0.0, priority=1)
        assert c_high < c_low

    def test_priority_offset_formula(self):
        """Cost difference between priorities = delta * priority_weight."""
        cfg = ScoringConfig(priority_weight=2.0)
        c5 = cost(0, 0.0, config=cfg, priority=5)
        c8 = cost(0, 0.0, config=cfg, priority=8)
        # priority 8 → offset (8-5)*2 = -6; priority 5 → offset 0
        assert c5 - c8 == pytest.approx((8 - 5) * 2.0)

    def test_default_priority_is_5(self):
        """Default priority of 5 matches explicit priority=5."""
        assert cost(0, 0.0) == pytest.approx(cost(0, 0.0, priority=5))


# ---------------------------------------------------------------------------
# Node priority — pick_node()
# ---------------------------------------------------------------------------

class TestPriorityPickNode:
    def test_high_priority_peer_wins(self):
        """A priority-9 peer beats a priority-5 self, all else equal."""
        peer = _alive_peer("b", queue_depth=0, rtt_ms=0.5, priority=9)
        table = _mock_table([peer])
        node, forwarded = pick_node("a", 0, _make_metrics(), table,
                                    _default_store(), self_priority=5)
        assert node == "b"
        assert forwarded is True

    def test_low_priority_peer_loses(self):
        """A priority-1 peer loses to a priority-9 self, all else equal."""
        peer = _alive_peer("b", queue_depth=0, rtt_ms=0.5, priority=1)
        table = _mock_table([peer])
        node, _ = pick_node("a", 0, _make_metrics(), table,
                            _default_store(), self_priority=9)
        assert node == "a"

    def test_equal_priority_uses_other_metrics(self):
        """Equal priority — scheduling still works based on queue/RTT."""
        peer = _alive_peer("b", queue_depth=5, rtt_ms=1.0, priority=5)
        table = _mock_table([peer])
        node, _ = pick_node("a", 0, _make_metrics(), table,
                            _default_store(), self_priority=5)
        assert node == "a"   # self wins on queue depth


# ---------------------------------------------------------------------------
# Congestion detection — RTTWindow
# ---------------------------------------------------------------------------

class TestCongestionDetection:
    def test_not_congested_with_few_samples(self):
        """Fewer than 3 samples → never congested (avoids false positives)."""
        w = RTTWindow()
        w.add(5.0)
        w.add(50.0)   # spike, but only 2 samples
        assert w.is_congested is False

    def test_not_congested_stable_rtt(self):
        """Consistent RTT → not congested."""
        w = RTTWindow()
        for _ in range(5):
            w.add(5.0)
        assert w.is_congested is False

    def test_congested_on_spike(self):
        """Latest RTT > 2× mean → congested."""
        w = RTTWindow()
        for _ in range(5):
            w.add(5.0)       # mean ≈ 5ms
        w.add(25.0)          # 25 > 5*2 → congested
        assert w.is_congested is True

    def test_not_congested_when_spike_clears(self):
        """Congestion clears when latest sample falls back to normal."""
        w = RTTWindow()
        for _ in range(5):
            w.add(5.0)
        w.add(25.0)          # congested
        w.add(5.0)           # recovers
        assert w.is_congested is False

    def test_congestion_score_higher_than_quality_score(self):
        """congestion_score > quality_score when congested."""
        w = RTTWindow()
        for _ in range(5):
            w.add(5.0)
        w.add(25.0)   # trigger congestion
        assert w.congestion_score(2.0, 3.0) > w.quality_score(2.0)

    def test_congestion_score_equals_quality_when_not_congested(self):
        """congestion_score == quality_score when not congested."""
        w = RTTWindow()
        for _ in range(6):
            w.add(5.0)
        assert w.congestion_score(2.0, 3.0) == pytest.approx(w.quality_score(2.0))

    def test_snapshot_includes_congested_flag(self):
        w = RTTWindow()
        snap = w.snapshot()
        assert "congested" in snap


# ---------------------------------------------------------------------------
# Congestion — pick_node()
# ---------------------------------------------------------------------------

class TestCongestionPickNode:
    def test_congested_peer_avoided(self):
        """A peer with a congested RTT window loses to a stable self."""
        window = RTTWindow()
        for _ in range(5):
            window.add(5.0)
        window.add(50.0)   # big spike → is_congested = True
        assert window.is_congested

        peer = _alive_peer("b", queue_depth=0, rtt_ms=5.0)
        table = _mock_table([peer], rtt_window=window)
        node, _ = pick_node("a", 0, _make_metrics(), table, _default_store())
        assert node == "a"


# ---------------------------------------------------------------------------
# AdaptiveController
# ---------------------------------------------------------------------------

def _make_controller(
    sampler_metrics: NodeMetrics,
    peers: list[PeerRecord] | None = None,
    rtt_window: RTTWindow | None = None,
) -> AdaptiveController:
    """Build a controller with injected metrics and peers."""
    store   = ConfigStore()
    sampler = MagicMock()
    sampler.latest = sampler_metrics
    table   = _mock_table(peers or [], rtt_window=rtt_window)
    return AdaptiveController(store, sampler, table, "a")


class TestAdaptiveControllerDetect:
    def test_balanced_by_default(self):
        ctrl = _make_controller(_make_metrics(cpu=20.0, power_source="ac"))
        assert ctrl._detect_target() == "balanced"

    def test_local_first_on_critical_battery(self):
        m = _make_metrics(power_source="battery", battery_percent=5.0)
        ctrl = _make_controller(m)
        assert ctrl._detect_target() == "local_first"

    def test_local_first_when_no_alive_peers(self):
        from datetime import timedelta
        dead = PeerRecord(
            node_id="b", host="127.0.0.1", port=9000,
            last_seen=datetime.now(timezone.utc) - timedelta(seconds=60),
        )
        m = _make_metrics(power_source="ac")
        ctrl = _make_controller(m, peers=[dead])
        assert ctrl._detect_target() == "local_first"

    def test_local_first_when_all_peers_congested(self):
        window = RTTWindow()
        for _ in range(5):
            window.add(5.0)
        window.add(50.0)   # congested
        peer = _alive_peer("b")
        ctrl = _make_controller(_make_metrics(), peers=[peer], rtt_window=window)
        assert ctrl._detect_target() == "local_first"

    def test_energy_first_on_low_battery(self):
        m = _make_metrics(power_source="battery", battery_percent=20.0)
        ctrl = _make_controller(m)
        assert ctrl._detect_target() == "energy_first"

    def test_latency_first_on_high_cpu(self):
        m = _make_metrics(cpu=90.0, power_source="ac")
        ctrl = _make_controller(m)
        assert ctrl._detect_target() == "latency_first"

    def test_battery_critical_beats_high_cpu(self):
        """Critical battery takes priority over high CPU."""
        m = _make_metrics(cpu=90.0, power_source="battery", battery_percent=5.0)
        ctrl = _make_controller(m)
        assert ctrl._detect_target() == "local_first"

    def test_ac_node_not_affected_by_battery_thresholds(self):
        """AC node with battery_percent set is still treated as AC."""
        m = _make_metrics(power_source="ac", battery_percent=5.0)
        ctrl = _make_controller(m)
        assert ctrl._detect_target() == "balanced"


class TestAdaptiveControllerHysteresis:
    def test_no_switch_on_first_detection(self):
        """Single detection should not trigger a switch (hysteresis)."""
        m = _make_metrics(cpu=90.0, power_source="ac")
        ctrl = _make_controller(m)
        result = ctrl.step()
        assert result is None
        assert ctrl.config_store.current.policy == "balanced"

    def test_switch_after_hysteresis_count(self):
        """Switch happens after HYSTERESIS_COUNT consecutive detections."""
        m = _make_metrics(cpu=90.0, power_source="ac")
        ctrl = _make_controller(m)
        result = None
        for _ in range(HYSTERESIS_COUNT):
            result = ctrl.step()
        assert result == "latency_first"
        assert ctrl.config_store.current.policy == "latency_first"

    def test_resets_on_condition_change(self):
        """Changing conditions before hysteresis is met resets the counter."""
        m_high_cpu = _make_metrics(cpu=90.0, power_source="ac")
        ctrl = _make_controller(m_high_cpu)
        ctrl.step()   # count=1 for latency_first

        # Now conditions change back to normal
        ctrl.sampler.latest = _make_metrics(cpu=10.0, power_source="ac")
        ctrl.step()   # candidate resets to "balanced", count=1

        # One more high-CPU reading — should not switch yet (count restarted)
        ctrl.sampler.latest = m_high_cpu
        result = ctrl.step()
        assert result is None   # only 1 tick for latency_first since reset

    def test_no_redundant_switch(self):
        """Already on the right policy → no switch even after threshold."""
        m = _make_metrics(cpu=90.0, power_source="ac")
        ctrl = _make_controller(m)
        ctrl.config_store.set_policy("latency_first")   # already set

        for _ in range(HYSTERESIS_COUNT + 2):
            result = ctrl.step()
        assert result is None   # policy already matches, no redundant switch
