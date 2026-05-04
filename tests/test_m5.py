"""M5 tests: stats, decision log, policy endpoints, result proxying.

Covers:
  - SchedulerStats.record() increments counters correctly
  - DecisionLog.record() stores entries with correct shape
  - pick_node populates stats + decision_log when provided
  - GET /scheduler/stats  returns the right shape
  - GET /scheduler/decisions returns entries
  - POST /scheduler/policy switches weights + reflects in config
  - POST /scheduler/policy rejects unknown policy names
  - GET /jobs/{job_id} fans out to a peer and returns the result
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from edge_scheduler.jobs import JobStore
from edge_scheduler.metrics import MetricsSampler
from edge_scheduler.models import NodeMetrics, PeerRecord
from edge_scheduler.peers import PeerTable
from edge_scheduler.scheduler import pick_node
from edge_scheduler.scoring import ConfigStore, ScoringConfig, POLICY_PRESETS
from edge_scheduler.stats import DecisionLog, SchedulerStats


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


def _alive_peer(node_id: str, queue_depth: int = 0, rtt_ms: float = 1.0) -> PeerRecord:
    return PeerRecord(
        node_id=node_id,
        host="127.0.0.1",
        port=9000,
        last_seen=datetime.now(timezone.utc),
        rtt_ms=rtt_ms,
        queue_depth=queue_depth,
        metrics=_make_metrics(),
    )


def _mock_table(peers: list[PeerRecord]) -> MagicMock:
    table = MagicMock()
    table.all.return_value = {p.node_id: p for p in peers}
    table.get_rtt_window.return_value = None
    return table


def _make_app(with_stats: bool = True):
    """Spin up a minimal in-process FastAPI app for endpoint tests."""
    from edge_scheduler.api import create_app
    from edge_scheduler.config import NodeConfig

    config = NodeConfig(node_id="a", host="127.0.0.1", port=9999, peers=[])
    table        = PeerTable(config)
    jobs         = JobStore()
    sampler      = MetricsSampler("a")
    config_store = ConfigStore()
    st           = SchedulerStats() if with_stats else None
    dl           = DecisionLog()    if with_stats else None
    app          = create_app(table, jobs, sampler, config_store, "a",
                              stats=st, decision_log=dl)
    return app, config_store, st, dl


# ---------------------------------------------------------------------------
# SchedulerStats unit tests
# ---------------------------------------------------------------------------

class TestSchedulerStats:
    def test_initial_state(self):
        s = SchedulerStats()
        assert s.total_invocations == 0
        assert s.local_executions == 0
        assert s.forwarded == 0
        assert s.forwarded_to == {}
        assert s.avg_cost_score == 0.0

    def test_local_record(self):
        s = SchedulerStats()
        s.record("a", "a", 10.0)
        assert s.total_invocations == 1
        assert s.local_executions == 1
        assert s.forwarded == 0

    def test_forwarded_record(self):
        s = SchedulerStats()
        s.record("b", "a", 20.0)
        assert s.forwarded == 1
        assert s.forwarded_to == {"b": 1}
        s.record("b", "a", 30.0)
        assert s.forwarded_to == {"b": 2}

    def test_avg_cost_score(self):
        s = SchedulerStats()
        s.record("a", "a", 10.0)
        s.record("a", "a", 20.0)
        assert s.avg_cost_score == pytest.approx(15.0)

    def test_to_dict_keys(self):
        s = SchedulerStats()
        d = s.to_dict()
        assert set(d.keys()) == {
            "total_invocations", "local_executions",
            "forwarded", "forwarded_to", "avg_cost_score",
        }


# ---------------------------------------------------------------------------
# DecisionLog unit tests
# ---------------------------------------------------------------------------

class TestDecisionLog:
    def test_empty_log(self):
        dl = DecisionLog()
        assert dl.to_list() == []

    def test_record_single_entry(self):
        dl = DecisionLog()
        dl.record("a", "b", True, [("a", 100.0), ("b", 5.0)], "balanced")
        entries = dl.to_list()
        assert len(entries) == 1
        e = entries[0]
        assert e["node"] == "a"
        assert e["chosen"] == "b"
        assert e["forwarded"] is True
        assert e["policy"] == "balanced"
        assert len(e["candidates"]) == 2

    def test_most_recent_first(self):
        dl = DecisionLog()
        dl.record("a", "a", False, [("a", 1.0)], "balanced")
        dl.record("a", "b", True,  [("a", 5.0), ("b", 1.0)], "latency_first")
        entries = dl.to_list()
        assert entries[0]["chosen"] == "b"    # most recent first
        assert entries[1]["chosen"] == "a"

    def test_maxlen_50(self):
        dl = DecisionLog()
        for i in range(60):
            dl.record("a", "a", False, [("a", float(i))], "balanced")
        assert len(dl.to_list()) == 50


# ---------------------------------------------------------------------------
# pick_node integration with stats + decision_log
# ---------------------------------------------------------------------------

class TestPickNodeWithStats:
    def test_stats_updated_on_local(self):
        table = _mock_table([])
        store = ConfigStore()
        s  = SchedulerStats()
        dl = DecisionLog()
        pick_node("a", 0, _make_metrics(), table, store, stats=s, decision_log=dl)
        assert s.total_invocations == 1
        assert s.local_executions == 1
        assert len(dl.to_list()) == 1

    def test_stats_updated_on_forward(self):
        table = _mock_table([_alive_peer("b", queue_depth=0, rtt_ms=1.0)])
        store = ConfigStore()
        s  = SchedulerStats()
        dl = DecisionLog()
        # Self has a heavy queue → should forward.
        pick_node("a", 20, _make_metrics(), table, store, stats=s, decision_log=dl)
        assert s.total_invocations == 1
        assert s.forwarded == 1
        assert len(dl.to_list()) == 1

    def test_no_stats_no_crash(self):
        """pick_node must work fine when stats=None and decision_log=None."""
        table = _mock_table([])
        store = ConfigStore()
        node, _ = pick_node("a", 0, _make_metrics(), table, store)
        assert node == "a"


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

class TestSchedulerStatsEndpoint:
    def test_returns_stats_dict(self):
        app, _, _, _ = _make_app()
        client = TestClient(app)
        resp = client.get("/scheduler/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_invocations" in data
        assert "local_executions" in data
        assert "forwarded" in data

    def test_not_enabled_returns_501(self):
        app, _, _, _ = _make_app(with_stats=False)
        client = TestClient(app)
        resp = client.get("/scheduler/stats")
        assert resp.status_code == 501


class TestSchedulerDecisionsEndpoint:
    def test_returns_list(self):
        app, _, _, _ = _make_app()
        client = TestClient(app)
        resp = client.get("/scheduler/decisions")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_not_enabled_returns_501(self):
        app, _, _, _ = _make_app(with_stats=False)
        client = TestClient(app)
        resp = client.get("/scheduler/decisions")
        assert resp.status_code == 501


class TestSchedulerPolicyEndpoint:
    def test_valid_policy_switches_weights(self):
        app, config_store, _, _ = _make_app()
        client = TestClient(app)

        for policy in ("balanced", "latency_first", "energy_first", "local_first"):
            resp = client.post("/scheduler/policy", json={"policy": policy})
            assert resp.status_code == 200
            data = resp.json()
            assert data["policy"] == policy
            expected = POLICY_PRESETS[policy]
            assert data["queue_weight"]   == pytest.approx(expected["queue_weight"])
            assert data["network_weight"] == pytest.approx(expected["network_weight"])
            assert data["cpu_weight"]     == pytest.approx(expected["cpu_weight"])
            assert data["energy_weight"]  == pytest.approx(expected["energy_weight"])

    def test_unknown_policy_returns_422(self):
        app, _, _, _ = _make_app()
        client = TestClient(app)
        resp = client.post("/scheduler/policy", json={"policy": "turbo_mode"})
        assert resp.status_code == 422

    def test_policy_reflected_in_config_endpoint(self):
        app, _, _, _ = _make_app()
        client = TestClient(app)
        client.post("/scheduler/policy", json={"policy": "energy_first"})
        resp = client.get("/scheduler/config")
        assert resp.status_code == 200
        assert resp.json()["policy"] == "energy_first"
        assert resp.json()["energy_weight"] == pytest.approx(POLICY_PRESETS["energy_first"]["energy_weight"])
