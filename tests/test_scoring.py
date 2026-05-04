"""Unit tests for M4 scoring: RTTWindow, ScoringConfig, cost(), ConfigStore."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from edge_scheduler.scoring import (
    ConfigStore,
    RTTWindow,
    ScoringConfig,
    cost,
)


# ---------------------------------------------------------------------------
# RTTWindow
# ---------------------------------------------------------------------------

class TestRTTWindow:
    def test_empty_mean_is_zero(self):
        w = RTTWindow()
        assert w.mean() == 0.0

    def test_empty_stddev_is_zero(self):
        w = RTTWindow()
        assert w.stddev() == 0.0

    def test_empty_quality_score_is_zero(self):
        w = RTTWindow()
        assert w.quality_score(variance_penalty=2.0) == 0.0

    def test_single_sample_mean(self):
        w = RTTWindow()
        w.add(10.0)
        assert w.mean() == pytest.approx(10.0)

    def test_single_sample_stddev_is_zero(self):
        w = RTTWindow()
        w.add(10.0)
        assert w.stddev() == 0.0

    def test_mean_of_multiple_samples(self):
        w = RTTWindow()
        for v in [2.0, 4.0, 6.0]:
            w.add(v)
        assert w.mean() == pytest.approx(4.0)

    def test_stddev_known_values(self):
        # population stddev of [2, 4, 6] = sqrt(((4+0+4)/3)) = sqrt(8/3)
        w = RTTWindow()
        for v in [2.0, 4.0, 6.0]:
            w.add(v)
        expected = math.sqrt(8.0 / 3.0)
        assert w.stddev() == pytest.approx(expected, rel=1e-4)

    def test_quality_score_includes_variance(self):
        w = RTTWindow()
        for v in [2.0, 4.0, 6.0]:
            w.add(v)
        penalty = 2.0
        expected = w.mean() + w.stddev() * penalty
        assert w.quality_score(penalty) == pytest.approx(expected)

    def test_window_rolls_over(self):
        from edge_scheduler.scoring import RTT_WINDOW_SIZE
        w = RTTWindow()
        # Fill beyond window size — early samples should be evicted.
        for i in range(RTT_WINDOW_SIZE + 5):
            w.add(float(i))
        assert len(w._samples) == RTT_WINDOW_SIZE

    def test_high_variance_higher_quality_score(self):
        stable = RTTWindow()
        for v in [5.0, 5.1, 4.9, 5.0]:
            stable.add(v)

        unstable = RTTWindow()
        for v in [1.0, 50.0, 1.0, 50.0]:
            unstable.add(v)

        assert unstable.quality_score(2.0) > stable.quality_score(2.0)

    def test_snapshot_has_expected_keys(self):
        w = RTTWindow()
        w.add(5.0)
        snap = w.snapshot()
        assert "mean_ms" in snap
        assert "stddev_ms" in snap
        assert "samples" in snap
        assert snap["samples"] == 1


# ---------------------------------------------------------------------------
# ScoringConfig
# ---------------------------------------------------------------------------

class TestScoringConfig:
    def test_defaults_are_set(self):
        cfg = ScoringConfig()
        assert cfg.queue_weight == 10.0
        assert cfg.network_weight == 1.0
        assert cfg.cpu_weight == 0.5
        assert cfg.energy_weight == 0.3
        assert cfg.variance_penalty == 2.0
        assert cfg.jitter_threshold == 5.0

    def test_custom_values(self):
        cfg = ScoringConfig(queue_weight=5.0, cpu_weight=1.0)
        assert cfg.queue_weight == 5.0
        assert cfg.cpu_weight == 1.0
        assert cfg.network_weight == 1.0   # default unchanged

    def test_to_dict_has_all_fields(self):
        cfg = ScoringConfig()
        d = cfg.to_dict()
        for key in ("queue_weight", "network_weight", "cpu_weight",
                    "energy_weight", "variance_penalty", "jitter_threshold",
                    "updated_at"):
            assert key in d


# ---------------------------------------------------------------------------
# cost()
# ---------------------------------------------------------------------------

class TestCostFunction:
    def _cfg(self, **kwargs) -> ScoringConfig:
        return ScoringConfig(**kwargs)

    def test_zero_everything(self):
        assert cost(0, 0.0, 0.0, 0.0, ScoringConfig()) == 0.0

    def test_queue_contribution(self):
        cfg = self._cfg(queue_weight=10.0)
        assert cost(3, 0.0, 0.0, 0.0, cfg) == pytest.approx(30.0)

    def test_network_contribution(self):
        cfg = self._cfg(network_weight=2.0)
        assert cost(0, 5.0, 0.0, 0.0, cfg) == pytest.approx(10.0)

    def test_cpu_contribution(self):
        cfg = self._cfg(cpu_weight=0.5)
        assert cost(0, 0.0, 80.0, 0.0, cfg) == pytest.approx(40.0)

    def test_energy_contribution(self):
        cfg = self._cfg(energy_weight=0.3)
        assert cost(0, 0.0, 0.0, 50.0, cfg) == pytest.approx(15.0)

    def test_all_combined(self):
        cfg = ScoringConfig(
            queue_weight=10.0, network_weight=1.0,
            cpu_weight=0.5, energy_weight=0.3
        )
        expected = (2*10.0) + (5.0*1.0) + (60.0*0.5) + (30.0*0.3)
        assert cost(2, 5.0, 60.0, 30.0, cfg) == pytest.approx(expected)

    def test_none_config_uses_defaults(self):
        default_cfg = ScoringConfig()
        assert cost(1, 1.0, 0.0, 0.0) == pytest.approx(
            cost(1, 1.0, 0.0, 0.0, default_cfg)
        )


# ---------------------------------------------------------------------------
# ConfigStore
# ---------------------------------------------------------------------------

class TestConfigStore:
    def test_init_with_default(self):
        store = ConfigStore()
        assert isinstance(store.current, ScoringConfig)

    def test_init_with_custom(self):
        cfg = ScoringConfig(queue_weight=99.0)
        store = ConfigStore(cfg)
        assert store.current.queue_weight == 99.0

    def test_update_local_always_applies(self):
        store = ConfigStore()
        new = ScoringConfig(cpu_weight=9.9)
        store.update_local(new)
        assert store.current.cpu_weight == 9.9

    def test_update_local_stamps_new_timestamp(self):
        store = ConfigStore()
        old_ts = store.current.updated_at
        import time; time.sleep(0.01)
        store.update_local(ScoringConfig())
        assert store.current.updated_at > old_ts

    def test_merge_gossip_adopts_newer(self):
        store = ConfigStore()
        newer = ScoringConfig(
            cpu_weight=7.7,
            updated_at=datetime.now(timezone.utc) + timedelta(seconds=10)
        )
        adopted = store.merge_gossip(newer)
        assert adopted is True
        assert store.current.cpu_weight == 7.7

    def test_merge_gossip_rejects_older(self):
        store = ConfigStore()
        original_weight = store.current.queue_weight
        older = ScoringConfig(
            queue_weight=99.0,
            updated_at=datetime.now(timezone.utc) - timedelta(seconds=60)
        )
        adopted = store.merge_gossip(older)
        assert adopted is False
        assert store.current.queue_weight == original_weight

    def test_merge_gossip_rejects_same_timestamp(self):
        store = ConfigStore()
        ts = store.current.updated_at
        same = ScoringConfig(cpu_weight=99.0, updated_at=ts)
        adopted = store.merge_gossip(same)
        assert adopted is False
