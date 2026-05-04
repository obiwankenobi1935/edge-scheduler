"""Load node configuration from a YAML file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from edge_scheduler.scoring import ScoringConfig


class PeerEntry(BaseModel):
    node_id: str
    host: str
    port: int


class NodeConfig(BaseModel):
    node_id: str
    host: str
    port: int
    peers: list[PeerEntry]
    scoring: ScoringConfig = ScoringConfig()   # M4: optional, defaults apply if absent


def load_config(path: str | Path) -> NodeConfig:
    """Read a YAML config file and return a validated NodeConfig."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    return NodeConfig(**raw)
