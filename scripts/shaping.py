"""Potential-based reward shaping for the Balatro PPO runs.

The env's built-in shaping already pays for blinds beaten, chip progress and
the terminal outcome. It pays nothing for the things that decide whether you
survive ante 4: money, jokers, and hand levels. This module adds those as a
*potential*, not as a bonus.

The distinction matters. A plain bonus ("+0.1 for buying a good joker")
changes the objective: the agent optimizes our joker ranking rather than
winning, and inherits our ceiling. Potential-based shaping instead pays

    F(s, s') = gamma * PHI(s') - PHI(s)

which telescopes to a constant over any trajectory, so the optimal policy is
provably unchanged (Ng, Harada & Russell 1999). It only redistributes reward
in time, turning "this pays off ten decisions later" into immediate signal.
If our joker ranking is wrong, the agent can still learn to ignore it -- the
worst case is a slower run, not a worse policy.

PHI is deliberately built from the joker's own numbers (rarity, xmult, mult,
chips) rather than a hand-curated list of 150 jokers, so it degrades to
"roughly right" for jokers nobody ranked.
"""

from __future__ import annotations

import math
from typing import Any

# Weights of the three potential terms. Scale is chosen against the env's own
# shaping, where beating a blind pays 0.15: a full joker slate is worth about
# one blind, so the agent is nudged, not bribed.
W_JOKER: float = 0.10
W_MONEY: float = 0.04
W_HAND_LEVEL: float = 0.03

_MONEY_REF: float = 25.0     # dollars where the money term saturates
_MAX_JOKERS: int = 5

# Rarity ordinals as jackdaw stores them: 1 common ... 4 legendary.
_RARITY_NORM: float = 4.0


def joker_quality(joker: Any) -> float:
    """Rough efficiency score for one joker, in [0, 1].

    Derived from the joker's own ability numbers so that unknown jokers still
    get a sane score:

    * xmult is the dominant scaling term in Balatro, so it dominates here;
    * flat mult and chips help early and are worth less;
    * rarity is a weak prior for "this does something strong".
    """
    ability = getattr(joker, "ability", None) or {}
    if isinstance(ability, dict):
        x_mult = float(ability.get("x_mult", 1.0) or 1.0)
        mult = float(ability.get("mult", 0.0) or 0.0)
        chips = float(ability.get("t_chips", ability.get("chips", 0.0)) or 0.0)
    else:
        x_mult = float(getattr(ability, "x_mult", 1.0) or 1.0)
        mult = float(getattr(ability, "mult", 0.0) or 0.0)
        chips = float(getattr(ability, "chips", 0.0) or 0.0)

    rarity = float(getattr(joker, "rarity", 1) or 1)

    q = 0.0
    q += 0.50 * min(max(x_mult - 1.0, 0.0) / 3.0, 1.0)   # x1 -> 0, x4 -> capped
    q += 0.20 * min(mult / 30.0, 1.0)
    q += 0.15 * min(chips / 150.0, 1.0)
    q += 0.15 * min(rarity / _RARITY_NORM, 1.0)
    return min(q, 1.0)


def potential(gs: dict[str, Any] | None) -> float:
    """PHI(s): a scalar "how healthy is this run" in roughly [0, 1].

    Terminal states must have potential 0 for the invariance argument to hold;
    callers pass ``None`` (or an empty state) there.
    """
    if not gs:
        return 0.0

    jokers = gs.get("jokers") or []
    joker_term = sum(joker_quality(j) for j in jokers) / _MAX_JOKERS

    dollars = float(gs.get("dollars", 0) or 0)
    # Log scale: going 0 -> 10 dollars matters far more than 60 -> 70.
    money_term = math.log1p(max(dollars, 0.0)) / math.log1p(_MONEY_REF)

    levels = gs.get("hand_levels")
    level_term = 0.0
    if levels is not None:
        try:
            # HandLevels keys its internal map by HandType; there is no public
            # iterator, so read the mapping directly rather than hardcoding the
            # 12 hand names.
            total = sum(max(st.level - 1, 0) for st in levels._hands.values())
            level_term = min(total / 10.0, 1.0)
        except Exception:
            level_term = 0.0

    return (
        W_JOKER * min(joker_term, 1.0)
        + W_MONEY * min(money_term, 1.0)
        + W_HAND_LEVEL * level_term
    )


class PotentialShaper:
    """Tracks PHI across a single env's trajectory and emits F(s, s')."""

    __slots__ = ("gamma", "coef", "_prev")

    def __init__(self, gamma: float, coef: float = 1.0) -> None:
        self.gamma = gamma
        self.coef = coef
        self._prev = 0.0

    def reset(self, gs: dict[str, Any] | None) -> None:
        self._prev = potential(gs)

    def step(self, gs: dict[str, Any] | None, done: bool) -> float:
        """Shaping term for the transition that just produced ``gs``."""
        phi_next = 0.0 if done else potential(gs)
        f = self.gamma * phi_next - self._prev
        self._prev = phi_next
        return self.coef * f
