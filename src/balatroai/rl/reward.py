"""Dense RL reward for balatroAI.

Three additive terms (see module docstring conventions elsewhere):
  1. score_delta = (state.round.chips - prev_state.round.chips) / 1000
  2. per-joker:  chips_added/1000 + mult_added/10 + log(xmult_factor)
  3. ante bonus: +ANTE_BONUS when state.ante_num > prev_state.ante_num
"""

from __future__ import annotations

import math

SCALE_CHIPS = 1.0 / 1000.0
SCALE_MULT = 0.1
SCALE_MONEY = 0.01  # money is a means, keep it small so it doesn't dwarf score
ANTE_BONUS = 5.0


def _chips(state: dict) -> float:
    if not state:
        return 0.0
    return float(state.get("round", {}).get("chips", 0.0))


def _money(state: dict) -> float:
    if not state:
        return 0.0
    return float(state.get("money", 0.0))


def _f(d: dict, key: str, default: float = 0.0) -> float:
    v = d.get(key, default)
    return float(v) if v is not None else float(default)


def _per_joker_term(last_score) -> float:
    per_joker = getattr(last_score, "per_joker", None) or {}
    total = 0.0
    for e in per_joker.values():
        total += _f(e, "chips_added") * SCALE_CHIPS
        total += _f(e, "mult_added") * SCALE_MULT
        total += math.log(max(1.0, _f(e, "xmult_factor", 1.0)))
    return total


def reward(prev_state: dict, state: dict, last_score) -> float:
    """Dense reward. last_score may be None (no hand yet); states optional."""
    score_delta = (_chips(state) - _chips(prev_state)) * SCALE_CHIPS
    money_delta = (_money(state) - _money(prev_state)) * SCALE_MONEY
    joker_term = _per_joker_term(last_score)
    prev_ante = prev_state.get("ante_num", 0) if prev_state else 0
    ante_term = ANTE_BONUS if state.get("ante_num", 0) > prev_ante else 0.0
    return score_delta + money_delta + joker_term + ante_term


if __name__ == "__main__":
    from collections import namedtuple

    FakeScore = namedtuple("FakeScore", "per_joker chips mult total",
                           defaults=({}, 0, 0, 0))
    prev0 = {"round": {"chips": 0.0}, "ante_num": 1}
    none = FakeScore(per_joker={})

    # no change -> 0
    assert abs(reward(prev0, dict(prev0), none)) < 1e-9

    # joker fires + score up -> positive
    fired = FakeScore(per_joker={"j_t": {"chips_added": 300, "mult_added": 5,
                                          "xmult_factor": 3.0}})
    s1 = {"round": {"chips": 2000.0}, "ante_num": 1}
    assert reward(prev0, s1, fired) > 0

    # ante milestone -> exactly bonus
    s2 = {"round": {"chips": 2000.0}, "ante_num": 2}
    assert reward(s1, s2, none) == ANTE_BONUS

    # edge cases
    assert reward(None, {"round": {}}, None) == 0.0
    assert reward(None, {"round": {"chips": 500.0}}, None) == 0.5

    print("PASS")