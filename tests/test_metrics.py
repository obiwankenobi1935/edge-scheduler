"""Unit tests for MetricsSampler and NodeMetrics."""

from datetime import datetime, timezone

import pytest

from edge_scheduler.metrics import MetricsSampler, _sample_now
from edge_scheduler.models import NodeMetrics


class TestSampleNow:
    def test_returns_node_metrics(self):
        result = _sample_now("a")
        assert isinstance(result, NodeMetrics)

    def test_node_id_set(self):
        result = _sample_now("test-node")
        assert result.node_id == "test-node"

    def test_cpu_in_valid_range(self):
        result = _sample_now("a")
        assert 0.0 <= result.cpu_percent <= 100.0

    def test_memory_in_valid_range(self):
        result = _sample_now("a")
        assert 0.0 <= result.memory_percent <= 100.0

    def test_memory_available_positive(self):
        result = _sample_now("a")
        assert result.memory_available_mb > 0

    def test_energy_proxy_in_valid_range(self):
        result = _sample_now("a")
        assert 0.0 <= result.energy_proxy <= 100.0

    def test_energy_proxy_formula(self):
        result = _sample_now("a")
        expected = result.cpu_percent * result.memory_percent / 100.0
        assert abs(result.energy_proxy - expected) < 0.1  # allow rounding

    def test_sampled_at_is_utc(self):
        result = _sample_now("a")
        assert result.sampled_at.tzinfo is not None


class TestMetricsSampler:
    def test_latest_is_populated_on_init(self):
        sampler = MetricsSampler("a")
        assert sampler.latest is not None
        assert isinstance(sampler.latest, NodeMetrics)

    def test_latest_node_id_matches(self):
        sampler = MetricsSampler("node-x")
        assert sampler.latest.node_id == "node-x"

    def test_to_dict_has_all_fields(self):
        sampler = MetricsSampler("a")
        d = sampler.latest.to_dict()
        for key in ("node_id", "cpu_percent", "memory_percent",
                    "memory_available_mb", "energy_proxy", "sampled_at"):
            assert key in d, f"Missing field: {key}"


class TestNodeMetricsModel:
    def _make(self, cpu=20.0, mem=50.0) -> NodeMetrics:
        return NodeMetrics(
            node_id="a",
            cpu_percent=cpu,
            memory_percent=mem,
            memory_available_mb=4096.0,
            energy_proxy=round(cpu * mem / 100, 2),
            sampled_at=datetime.now(timezone.utc),
        )

    def test_high_cpu_high_energy(self):
        m = self._make(cpu=90.0, mem=80.0)
        assert m.energy_proxy == pytest.approx(90.0 * 80.0 / 100.0, abs=0.1)

    def test_zero_cpu_zero_energy(self):
        m = self._make(cpu=0.0, mem=50.0)
        assert m.energy_proxy == 0.0
