"""Load personality.yaml into a validated model."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Personality(BaseModel):
    name: str
    character: str
    voice: str = ""
    rules: list[str] = Field(default_factory=list)
    greeting: str = ""


def load_personality(path: str | Path) -> Personality:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Personality(**data)
