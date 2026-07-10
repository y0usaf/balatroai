"""Bot registry — the one declaration mechanism for bots (doctrine 05).

A bot is a class with `name: str` and `act(state: dict) -> Action`.
Bots read immutable gamestate snapshots and return queued actions; they
never talk to the server directly (doctrine 02).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Action:
    method: str
    params: dict = field(default_factory=dict)
    note: str = ""


class Bot(Protocol):
    name: str

    def act(self, state: dict) -> Action: ...


BOTS: dict[str, type] = {}


def register(cls: type) -> type:
    BOTS[cls.name] = cls
    return cls


def get_bot(name: str) -> Bot:
    # Import here so the registry is populated exactly once, on first use.
    from . import heuristic, random_bot  # noqa: F401

    if name not in BOTS:
        raise KeyError(f"unknown bot {name!r}; available: {', '.join(sorted(BOTS))}")
    return BOTS[name]()
