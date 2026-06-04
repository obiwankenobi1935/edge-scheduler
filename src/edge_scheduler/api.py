"""FastAPI application: endpoints for gossip, invocation, scoring, and debugging.

M5 additions:
  - create_app now accepts SchedulerStats and DecisionLog
  - pick_node calls now record stats + decisions
  - GET /jobs/{job_id} fans out to peers when the job isn't found locally
  - GET /scheduler/stats   — running counters
  - GET /scheduler/decisions — last-N scheduling decisions
  - POST /scheduler/policy  — switch named policy preset
"""

from __future__ import annotations

import asyncio

import httpx
import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from edge_scheduler.adaptive import AdaptiveController
from edge_scheduler.jobs import JobStore
from edge_scheduler.metrics import MetricsSampler
from edge_scheduler.models import ExecuteRequest, InvokeRequest, InvokeResponse, PeerUpdate
from edge_scheduler.peers import PeerTable
from edge_scheduler.scheduler import pick_node
from edge_scheduler.scoring import ConfigStore, Policy, ScoringConfig
from edge_scheduler.stats import DecisionLog, SchedulerStats

log = structlog.get_logger()

REQUEST_TIMEOUT_S = 10.0


def create_app(
    table: PeerTable,
    jobs: JobStore,
    sampler: MetricsSampler,
    config_store: ConfigStore,      # M4
    node_id: str,
    stats: SchedulerStats | None = None,              # M5
    decision_log: DecisionLog | None = None,          # M5
    self_priority: int = 5,                           # M6
    adaptive: AdaptiveController | None = None,       # M6
) -> FastAPI:
    app = FastAPI(title=f"edge-scheduler node {node_id}")

    # ------------------------------------------------------------------
    # M1 endpoints: gossip + debug
    # ------------------------------------------------------------------

    @app.post("/peer/update")
    async def peer_update(body: PeerUpdate) -> dict:
        table.update_last_seen(
            body.from_node_id,
            queue_depth=body.queue_depth,
            metrics=body.metrics,
            priority=body.priority,            # M6
        )
        # M4: adopt gossiped config if it is newer than ours.
        if body.scoring_config is not None:
            config_store.merge_gossip(body.scoring_config)
        return {"ok": True}

    @app.get("/peers")
    async def peers() -> JSONResponse:
        return JSONResponse(content=table.snapshot())

    @app.get("/health")
    async def health() -> dict:
        return {"node_id": node_id, "status": "ok"}

    # ------------------------------------------------------------------
    # M3 endpoint: local metrics
    # ------------------------------------------------------------------

    @app.get("/metrics")
    async def metrics() -> JSONResponse:
        return JSONResponse(content=sampler.latest.to_dict())

    # ------------------------------------------------------------------
    # M4 endpoints: scoring config inspection + hot-reload
    # ------------------------------------------------------------------

    @app.get("/scheduler/config")
    async def get_scheduler_config() -> JSONResponse:
        return JSONResponse(content=config_store.current.to_dict())

    @app.post("/scheduler/config")
    async def post_scheduler_config(body: ScoringConfig) -> JSONResponse:
        """Hot-reload weights. Change propagates to all peers within 2 seconds."""
        config_store.update_local(body)
        log.info("scoring_config_hot_reloaded", node=node_id)
        return JSONResponse(content=config_store.current.to_dict())

    # ------------------------------------------------------------------
    # M2 endpoints: invocation
    # ------------------------------------------------------------------

    @app.post("/invoke", response_model=InvokeResponse)
    async def invoke(body: InvokeRequest) -> InvokeResponse:
        chosen, forwarded = pick_node(
            self_node_id=node_id,
            self_queue_depth=jobs.queue_depth,
            self_metrics=sampler.latest,
            table=table,
            config_store=config_store,
            stats=stats,                    # M5
            decision_log=decision_log,      # M5
            self_priority=self_priority,    # M6
        )

        if not forwarded:
            job = jobs.create(
                function=body.function,
                args=body.args,
                scheduled_on=node_id,
                origin_node=node_id,
            )
            asyncio.create_task(jobs.run(job))
            log.info("invoke_local", job_id=job.job_id, function=body.function)
            return InvokeResponse(job_id=job.job_id, scheduled_on=node_id, forwarded=False)

        peer = table.get(chosen)
        if peer is None:
            raise HTTPException(status_code=503, detail=f"Peer {chosen} not found")

        job = jobs.create(
            function=body.function,
            args=body.args,
            scheduled_on=chosen,
            origin_node=node_id,
        )

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://{peer.host}:{peer.port}/execute",
                    json=ExecuteRequest(
                        job_id=job.job_id,
                        function=body.function,
                        args=body.args,
                        origin_node=node_id,
                    ).model_dump(),
                    timeout=REQUEST_TIMEOUT_S,
                )
                resp.raise_for_status()
        except Exception as exc:
            log.warning("forward_failed", peer=chosen, error=str(exc), job_id=job.job_id)
            job.scheduled_on = node_id
            asyncio.create_task(jobs.run(job))
            return InvokeResponse(job_id=job.job_id, scheduled_on=node_id, forwarded=False)

        log.info("invoke_forwarded", job_id=job.job_id, function=body.function, peer=chosen)
        return InvokeResponse(job_id=job.job_id, scheduled_on=chosen, forwarded=True)

    @app.post("/execute")
    async def execute(body: ExecuteRequest) -> dict:
        job = jobs.create(
            function=body.function,
            args=body.args,
            scheduled_on=node_id,
            origin_node=body.origin_node,
            job_id=body.job_id,
        )
        asyncio.create_task(jobs.run(job))
        log.info("execute_received", job_id=job.job_id, function=body.function,
                 from_node=body.origin_node)
        return {"ok": True, "job_id": job.job_id}

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str) -> JSONResponse:
        # Check locally first.
        job = jobs.get(job_id)
        if job is not None:
            return JSONResponse(content=job.to_dict())

        # M5: fan out to alive peers in parallel.
        alive_peers = [p for p in table.all().values() if p.status() == "alive"]

        async def _fetch_from_peer(peer) -> dict | None:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"http://{peer.host}:{peer.port}/jobs/{job_id}",
                        timeout=REQUEST_TIMEOUT_S,
                    )
                    if resp.status_code == 200:
                        return resp.json()
            except Exception:
                pass
            return None

        results = await asyncio.gather(*[_fetch_from_peer(p) for p in alive_peers])
        for result in results:
            if result is not None:
                return JSONResponse(content=result)

        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    @app.get("/queue")
    async def queue() -> JSONResponse:
        return JSONResponse(content={
            "node_id": node_id,
            "queue_depth": jobs.queue_depth,
            "jobs": [j.to_dict() for j in jobs.all()],
        })

    # ------------------------------------------------------------------
    # M5 endpoints: stats, decision log, policy switching
    # ------------------------------------------------------------------

    @app.get("/scheduler/stats")
    async def scheduler_stats() -> JSONResponse:
        """Running counters: invocations, local vs forwarded, avg cost score."""
        if stats is None:
            return JSONResponse(content={"detail": "stats not enabled"}, status_code=501)
        return JSONResponse(content=stats.to_dict())

    @app.get("/scheduler/decisions")
    async def scheduler_decisions() -> JSONResponse:
        """Last-50 scheduling decisions with candidate costs and chosen node."""
        if decision_log is None:
            return JSONResponse(content={"detail": "decision log not enabled"}, status_code=501)
        return JSONResponse(content=decision_log.to_list())

    @app.get("/scheduler/mode")
    async def scheduler_mode() -> JSONResponse:
        """Show what the adaptive controller is currently detecting and why."""
        m     = sampler.latest
        peers = list(table.all().values())
        alive = [p for p in peers if p.status() == "alive"]
        return JSONResponse(content={
            "node_id":        node_id,
            "active_policy":  config_store.current.policy,
            "adaptive":       adaptive is not None,
            "conditions": {
                "power_source":       m.power_source,
                "battery_percent":    m.battery_percent,
                "cpu_percent":        m.cpu_percent,
                "alive_peers":        len(alive),
                "congested_peers":    sum(
                    1 for p in alive
                    if (w := table.get_rtt_window(p.node_id)) and w.is_congested
                ),
            },
            "target_policy": adaptive._detect_target() if adaptive else None,
            "reason":        adaptive._detect_reason() if adaptive else "manual",
        })

    @app.post("/scheduler/policy")
    async def set_policy(body: dict) -> JSONResponse:
        """Switch to a named policy preset (balanced/latency_first/energy_first/local_first).

        Body: {"policy": "<name>"}
        The new config is picked up by gossip within 2 s and propagates to all peers.
        """
        policy_name = body.get("policy")
        valid: list[Policy] = ["balanced", "latency_first", "energy_first", "local_first"]
        if policy_name not in valid:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown policy '{policy_name}'. Valid options: {valid}",
            )
        config_store.set_policy(policy_name)          # type: ignore[arg-type]
        log.info("policy_set_via_api", policy=policy_name, node=node_id)
        return JSONResponse(content=config_store.current.to_dict())

    # ------------------------------------------------------------------
    # Dashboard — self-contained HTML page, polls the existing API
    # ------------------------------------------------------------------

    @app.get("/dashboard", response_class=None)
    async def dashboard():
        from fastapi.responses import HTMLResponse
        html = _DASHBOARD_HTML.replace("__NODE_ID__", node_id)
        return HTMLResponse(content=html)

    return app


# ---------------------------------------------------------------------------
# Dashboard HTML (self-contained — no external dependencies)
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Edge Scheduler — __NODE_ID__</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }

  /* Header */
  .header { background: #1a1d2e; border-bottom: 1px solid #2d3748; padding: 16px 24px;
            display: flex; align-items: center; justify-content: space-between; }
  .header h1 { font-size: 1.2rem; font-weight: 600; color: #a78bfa; letter-spacing: 0.05em; }
  .header .subtitle { font-size: 0.8rem; color: #64748b; margin-top: 2px; }
  .header .badge { font-size: 0.75rem; padding: 4px 12px; border-radius: 20px;
                   background: #1e293b; border: 1px solid #334155; color: #94a3b8; }
  .badge.policy-balanced    { border-color: #3b82f6; color: #60a5fa; }
  .badge.policy-latency_first { border-color: #f59e0b; color: #fbbf24; }
  .badge.policy-energy_first  { border-color: #10b981; color: #34d399; }
  .badge.policy-local_first   { border-color: #ef4444; color: #f87171; }

  /* Layout */
  .main { padding: 20px 24px; display: grid; gap: 20px; }
  .row { display: grid; gap: 16px; }
  .row-3 { grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
  .row-2 { grid-template-columns: 1fr 1fr; }

  /* Cards */
  .card { background: #1a1d2e; border: 1px solid #2d3748; border-radius: 10px; padding: 16px; }
  .card-title { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
                color: #64748b; margin-bottom: 12px; }

  /* Node cards */
  .node-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }
  .node-id { font-size: 1.4rem; font-weight: 700; color: #e2e8f0; }
  .node-status { font-size: 0.72rem; font-weight: 600; padding: 3px 10px; border-radius: 12px; }
  .status-alive  { background: #064e3b; color: #34d399; border: 1px solid #065f46; }
  .status-stale  { background: #451a03; color: #fb923c; border: 1px solid #7c2d12; }
  .status-dead   { background: #450a0a; color: #f87171; border: 1px solid #7f1d1d; }
  .status-self   { background: #1e1b4b; color: #a78bfa; border: 1px solid #3730a3; }

  .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .metric { background: #0f1117; border-radius: 6px; padding: 8px 10px; }
  .metric-label { font-size: 0.65rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
  .metric-value { font-size: 1.0rem; font-weight: 600; color: #e2e8f0; margin-top: 2px; }
  .metric-value.warn  { color: #fb923c; }
  .metric-value.crit  { color: #f87171; }
  .metric-value.good  { color: #34d399; }

  .bar-wrap { margin-top: 10px; }
  .bar-label { font-size: 0.65rem; color: #64748b; display: flex; justify-content: space-between; margin-bottom: 3px; }
  .bar-track { background: #0f1117; border-radius: 4px; height: 5px; overflow: hidden; }
  .bar-fill  { height: 100%; border-radius: 4px; transition: width 0.5s ease; }
  .bar-cpu   { background: #6366f1; }
  .bar-queue { background: #f59e0b; }
  .bar-bat   { background: #10b981; }

  .node-footer { margin-top: 10px; font-size: 0.68rem; color: #475569;
                 display: flex; justify-content: space-between; }

  /* Stats row */
  .stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
  .stat-box  { background: #0f1117; border-radius: 8px; padding: 12px; text-align: center; }
  .stat-num  { font-size: 1.6rem; font-weight: 700; color: #a78bfa; }
  .stat-lbl  { font-size: 0.65rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px; }

  /* Decisions feed */
  .feed { max-height: 340px; overflow-y: auto; display: flex; flex-direction: column; gap: 6px; }
  .feed::-webkit-scrollbar { width: 4px; }
  .feed::-webkit-scrollbar-track { background: #0f1117; }
  .feed::-webkit-scrollbar-thumb { background: #334155; border-radius: 2px; }

  .decision { background: #0f1117; border-radius: 6px; padding: 10px 12px;
              border-left: 3px solid #334155; font-size: 0.78rem; }
  .decision.forwarded { border-left-color: #6366f1; }
  .decision.local     { border-left-color: #10b981; }
  .decision-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
  .decision-node { font-weight: 600; color: #e2e8f0; }
  .decision-tag  { font-size: 0.65rem; padding: 2px 7px; border-radius: 10px; }
  .tag-local    { background: #064e3b; color: #34d399; }
  .tag-forward  { background: #1e1b4b; color: #a78bfa; }
  .candidates   { display: flex; flex-wrap: wrap; gap: 5px; }
  .candidate    { background: #1a1d2e; border-radius: 4px; padding: 2px 7px; font-size: 0.65rem;
                  color: #94a3b8; border: 1px solid #2d3748; }
  .candidate.winner { border-color: #6366f1; color: #a78bfa; }
  .decision-time { font-size: 0.62rem; color: #475569; }

  /* Mode card */
  .mode-reason { font-size: 0.85rem; color: #94a3b8; margin-top: 8px; line-height: 1.5; }
  .mode-conditions { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 10px; }
  .cond-item { background: #0f1117; border-radius: 6px; padding: 7px 10px; font-size: 0.72rem; }
  .cond-key  { color: #64748b; margin-bottom: 2px; }
  .cond-val  { color: #e2e8f0; font-weight: 600; }

  /* Pulse dot */
  .pulse { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
           background: #34d399; margin-right: 6px; animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.3; } }

  .error-msg { color: #f87171; font-size: 0.8rem; padding: 8px; background: #450a0a;
               border-radius: 6px; margin-top: 8px; }
  .refresh-ts { font-size: 0.65rem; color: #475569; }

  @media (max-width: 768px) { .row-2 { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1><span class="pulse"></span>Edge Scheduler Dashboard</h1>
    <div class="subtitle">Connected to node <strong>__NODE_ID__</strong></div>
  </div>
  <div style="display:flex;gap:10px;align-items:center;">
    <span class="refresh-ts" id="last-refresh">—</span>
    <span class="badge" id="policy-badge">balanced</span>
  </div>
</div>

<div class="main">

  <!-- Node cards -->
  <div>
    <div class="card-title" style="margin-bottom:10px;">Cluster Nodes</div>
    <div class="row row-3" id="nodes-grid">
      <div class="card"><div class="card-title">Loading…</div></div>
    </div>
  </div>

  <!-- Stats -->
  <div class="card">
    <div class="card-title">Scheduling Stats</div>
    <div class="stat-grid" id="stats-grid">
      <div class="stat-box"><div class="stat-num" id="s-total">—</div><div class="stat-lbl">Total Jobs</div></div>
      <div class="stat-box"><div class="stat-num" id="s-local">—</div><div class="stat-lbl">Local</div></div>
      <div class="stat-box"><div class="stat-num" id="s-fwd">—</div><div class="stat-lbl">Forwarded</div></div>
      <div class="stat-box"><div class="stat-num" id="s-cost">—</div><div class="stat-lbl">Avg Cost</div></div>
    </div>
  </div>

  <!-- Decisions + Mode -->
  <div class="row row-2">
    <div class="card">
      <div class="card-title">Live Decision Feed</div>
      <div class="feed" id="decision-feed">
        <div style="color:#475569;font-size:0.8rem;">Waiting for decisions…</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Adaptive Controller</div>
      <div id="mode-content">
        <div style="color:#475569;font-size:0.8rem;">Loading…</div>
      </div>
    </div>
  </div>

</div>

<script>
const BASE = '';  // same origin

function pct(v, warn=70, crit=90) {
  const cls = v >= crit ? 'crit' : v >= warn ? 'warn' : 'good';
  return `<span class="metric-value ${cls}">${v.toFixed(1)}%</span>`;
}
function barColor(v, type) { return type; }
function timeAgo(iso) {
  const d = new Date(iso), now = new Date();
  const s = Math.round((now - d) / 1000);
  if (s < 5)  return 'just now';
  if (s < 60) return `${s}s ago`;
  return `${Math.round(s/60)}m ago`;
}
function policyClass(p) {
  return 'policy-' + (p || 'balanced').replace(/ /g,'_');
}

async function fetchJSON(path) {
  const r = await fetch(BASE + path);
  if (!r.ok) throw new Error(r.status);
  return r.json();
}

// ---- Render nodes ----
function renderNodes(peers) {
  const grid = document.getElementById('nodes-grid');
  if (!peers || Object.keys(peers).length === 0) {
    grid.innerHTML = '<div class="card" style="color:#475569">No peers visible yet</div>';
    return;
  }
  grid.innerHTML = Object.entries(peers).map(([id, p]) => {
    const st = p.status || 'unknown';
    const stCls = st === 'alive' ? 'status-alive' : st === 'stale' ? 'status-stale' : 'status-dead';
    const m = p.metrics || {};
    const cpu = m.cpu_percent ?? 0;
    const mem = m.memory_percent ?? 0;
    const q   = p.queue_depth ?? 0;
    const bat = m.battery_percent;
    const pwr = m.power_source ?? 'ac';
    const rtt = p.rtt_ms != null ? p.rtt_ms.toFixed(1) + ' ms' : '—';
    const policy = p.scoring_config?.policy ?? '—';
    const batShow = bat != null ? bat.toFixed(0) + '%' : 'N/A';
    const batCls  = bat != null && bat < 15 ? 'crit' : bat != null && bat < 30 ? 'warn' : 'good';
    const cpuCls  = cpu > 90 ? 'crit' : cpu > 70 ? 'warn' : 'good';

    return `<div class="card">
      <div class="node-header">
        <div class="node-id">node ${id}</div>
        <span class="node-status ${stCls}">${st}</span>
      </div>
      <div class="metrics">
        <div class="metric">
          <div class="metric-label">CPU</div>
          <div class="metric-value ${cpuCls}">${cpu.toFixed(1)}%</div>
        </div>
        <div class="metric">
          <div class="metric-label">Queue</div>
          <div class="metric-value ${q > 5 ? 'warn' : 'good'}">${q} jobs</div>
        </div>
        <div class="metric">
          <div class="metric-label">Memory</div>
          <div class="metric-value">${mem.toFixed(1)}%</div>
        </div>
        <div class="metric">
          <div class="metric-label">Battery</div>
          <div class="metric-value ${batCls}">${pwr === 'battery' ? '🔋' : '⚡'} ${batShow}</div>
        </div>
      </div>
      <div class="bar-wrap">
        <div class="bar-label"><span>CPU</span><span>${cpu.toFixed(0)}%</span></div>
        <div class="bar-track"><div class="bar-fill bar-cpu" style="width:${Math.min(cpu,100)}%"></div></div>
      </div>
      <div class="bar-wrap" style="margin-top:6px">
        <div class="bar-label"><span>Queue</span><span>${q}</span></div>
        <div class="bar-track"><div class="bar-fill bar-queue" style="width:${Math.min(q*10,100)}%"></div></div>
      </div>
      <div class="node-footer">
        <span>RTT: ${rtt}</span>
        <span>Policy: ${policy}</span>
      </div>
    </div>`;
  }).join('');
}

// ---- Render decisions ----
function renderDecisions(decisions) {
  const feed = document.getElementById('decision-feed');
  if (!decisions || decisions.length === 0) {
    feed.innerHTML = '<div style="color:#475569;font-size:0.8rem;">No decisions yet</div>';
    return;
  }
  feed.innerHTML = [...decisions].reverse().slice(0, 15).map(d => {
    const fwd = d.forwarded;
    const cands = Object.entries(d.candidates || {}).map(([nid, score]) => {
      const win = nid === d.chosen;
      return `<span class="candidate ${win ? 'winner' : ''}">${nid}: ${parseFloat(score).toFixed(1)}</span>`;
    }).join('');
    return `<div class="decision ${fwd ? 'forwarded' : 'local'}">
      <div class="decision-top">
        <span class="decision-node">→ ${d.chosen}</span>
        <span class="decision-tag ${fwd ? 'tag-forward' : 'tag-local'}">${fwd ? 'forwarded' : 'local'}</span>
      </div>
      <div class="candidates">${cands}</div>
      <div class="decision-time" style="margin-top:5px">${timeAgo(d.decided_at)} · policy: ${d.policy || '—'}</div>
    </div>`;
  }).join('');
}

// ---- Render mode ----
function renderMode(mode) {
  const el = document.getElementById('mode-content');
  if (!mode) { el.innerHTML = '<div class="error-msg">Unavailable</div>'; return; }
  const c = mode.conditions || {};
  el.innerHTML = `
    <div style="margin-bottom:8px">
      <span class="badge ${policyClass(mode.active_policy)}" style="font-size:0.8rem;padding:5px 14px;">
        ${mode.active_policy || '—'}
      </span>
    </div>
    <div class="mode-reason">${mode.reason || '—'}</div>
    <div class="mode-conditions">
      <div class="cond-item"><div class="cond-key">Power</div><div class="cond-val">${c.power_source ?? '—'}</div></div>
      <div class="cond-item"><div class="cond-key">Battery</div><div class="cond-val">${c.battery_percent != null ? c.battery_percent.toFixed(0)+'%' : 'N/A'}</div></div>
      <div class="cond-item"><div class="cond-key">CPU</div><div class="cond-val">${c.cpu_percent != null ? c.cpu_percent.toFixed(1)+'%' : '—'}</div></div>
      <div class="cond-item"><div class="cond-key">Alive peers</div><div class="cond-val">${c.alive_peers ?? '—'}</div></div>
      <div class="cond-item"><div class="cond-key">Congested</div><div class="cond-val">${c.congested_peers ?? 0} peer(s)</div></div>
      <div class="cond-item"><div class="cond-key">Target</div><div class="cond-val">${mode.target_policy || '—'}</div></div>
    </div>`;

  // Update header badge
  const badge = document.getElementById('policy-badge');
  badge.textContent = mode.active_policy || 'balanced';
  badge.className = 'badge ' + policyClass(mode.active_policy);
}

// ---- Render stats ----
function renderStats(stats) {
  if (!stats || stats.detail) return;
  document.getElementById('s-total').textContent = stats.total_invocations ?? '—';
  document.getElementById('s-local').textContent = stats.local_executions ?? '—';
  document.getElementById('s-fwd').textContent   = stats.total_forwarded ?? '—';
  const avg = stats.avg_cost_score;
  document.getElementById('s-cost').textContent  = avg != null ? avg.toFixed(1) : '—';
}

// ---- Main refresh loop ----
async function refresh() {
  try {
    const [peers, stats, decisions, mode] = await Promise.all([
      fetchJSON('/peers'),
      fetchJSON('/scheduler/stats').catch(() => null),
      fetchJSON('/scheduler/decisions').catch(() => []),
      fetchJSON('/scheduler/mode').catch(() => null),
    ]);
    renderNodes(peers);
    renderStats(stats);
    renderDecisions(decisions);
    renderMode(mode);
    document.getElementById('last-refresh').textContent =
      'Updated ' + new Date().toLocaleTimeString();
  } catch (e) {
    document.getElementById('last-refresh').textContent = 'Error: ' + e.message;
  }
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>"""
