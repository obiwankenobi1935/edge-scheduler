# Edge Scheduler

Decentralised multi-metric scheduler for serverless edge computing.

## M1 — Peer Discovery & Liveness

Three nodes start up, find each other via static YAML config, gossip every 2 seconds,
and mark a peer dead 6 seconds after it stops responding.

### Quick start

```powershell
# Terminal 1
pwsh ./scripts/run_node.ps1 a

# Terminal 2
pwsh ./scripts/run_node.ps1 b

# Terminal 3
pwsh ./scripts/run_node.ps1 c
```

Then check peers:

```
curl http://localhost:8001/peers
```

### Running tests

```bash
pip install -e ".[dev]"
pytest
```
