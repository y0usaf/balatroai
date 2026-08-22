"""Dense RL reward for balatroAI.

Potential-shaped additive terms:
  1. score_delta = capped chips progress toward the CURRENT blind / 1000.
     Chips past the blind target earn nothing — overkill farming is dead
     weight, and the cap makes "seal the blind cheaply" the optimal shape.
     Shaping is gated on staying within one blind (ante_num + round_num):
     across a transition the chip counter resets, and capping both sides
     would read as a large negative.
  2. per-joker: mult_added/10 + log(xmult_factor).  chips_added already
     lands in score_delta; counting it twice biased toward chip jokers.
  3. money_delta * 0.01 (a means, kept small).
  4. ante bonus: ANTE_BONUS * new_ante on ante-up — later antes are
     exponentially harder, so their milestone is worth more.
  5. terminal: +WIN_BONUS on a win, -LOSS_PENALTY on a loss, no shaping
     noise on the final step.

VecNormalize(norm_reward=True) absorbs the scale drift from the scaled
ante bonus.
"""

from __future__ import annotations

import math

SCALE_CHIPS = 1.0 / 1000.0
SCALE_MULT = 0.1
SCALE_MONEY = 0.01  # money is a means, keep it small so it doesn't dwarf score
ANTE_BONUS = 5.0
WIN_BONUS = 50.0
LOSS_PENALTY = 20.0


def _money(state: dict | None) -> float:
    if not state:
        return 0.0
    return float(state.get("money", 0.0))


def _f(d: dict, key: str, default: float = 0.0) -> float:
    v = d.get(key, default)
    return float(v) if v is not None else float(default)


def blind_target(state: dict | None) -> float:
    """Score needed to beat the CURRENT blind (0 if unknown)."""
    for blind in (state.get("blinds") or {}).values() if state else ():
        if isinstance(blind, dict) and blind.get("status") == "CURRENT":
            return float(blind.get("score", 0) or 0)
    return 0.0


def _progress(state: dict | None) -> float:
    """Chips that count toward the current blind (capped at its target)."""
    if not state:
        return 0.0
    chips = float(state.get("round", {}).get("chips", 0.0))
    target = blind_target(state)
    return min(chips, target) if target > 0 else chips


def _per_joker_term(last_score) -> float:
    per_joker = getattr(last_score, "per_joker", None) or {}
    total = 0.0
    for e in per_joker.values():
        total += _f(e, "mult_added") * SCALE_MULT
        total += math.log(max(1.0, _f(e, "xmult_factor", 1.0)))
    return total


def reward(prev_state: dict | None, state: dict, last_score) -> float:
    """Dense reward. last_score may be None (no hand yet); states optional."""
    if state.get("state") == "GAME_OVER":
        return WIN_BONUS if state.get("won") else -LOSS_PENALTY

    money_delta = (_money(state) - _money(prev_state)) * SCALE_MONEY
    joker_term = _per_joker_term(last_score)

    prev_ante = prev_state.get("ante_num", 0) if prev_state else 0
    ante_now = state.get("ante_num", 0)
    ante_term = ANTE_BONUS * ante_now if ante_now > prev_ante else 0.0

    same_blind = bool(prev_state) and (
        prev_state.get("ante_num"), prev_state.get("round_num"),
    ) == (state.get("ante_num"), state.get("round_num"))
    score_delta = (_progress(state) - _progress(prev_state)) * SCALE_CHIPS \
        if same_blind else 0.0

    return score_delta + money_delta + joker_term + ante_term


if __name__ == "__main__":
    from collections import namedtuple

    FakeScore = namedtuple("FakeScore", "per_joker chips mult total",
                           defaults=({}, 0, 0, 0))
    none = FakeScore(per_joker={})

    blind1 = {"small": {"status": "CURRENT", "score": 300}}

    # no change -> 0
    s0 = {"round": {"chips": 0.0}, "ante_num": 1, "round_num": 1, "blinds": blind1}
    assert abs(reward(s0, dict(s0), none)) < 1e-9

    # capped progress: chips toward target count, overkill does not
    half = {"round": {"chips": 150.0}, "ante_num": 1, "round_num": 1, "blinds": blind1}
    sealed = {"round": {"chips": 900.0}, "ante_num": 1, "round_num": 1, "blinds": blind1}
    assert abs(reward(s0, half, none) - 150 * SCALE_CHIPS) < 1e-9
    assert abs(reward(half, sealed, none) - 150 * SCALE_CHIPS) < 1e-9  # capped at target

    # blind transition resets the counter without a negative spike
    next_blind = {"round": {"chips": 0.0}, "ante_num": 1, "round_num": 2,
                  "blinds": {"big": {"status": "CURRENT", "score": 600}}}
    assert reward(sealed, next_blind, none) == 0.0

    # ante milestone scales with the ante reached
    a2 = {"round": {"chips": 0.0}, "ante_num": 2, "round_num": 3, "blinds": blind1}
    assert reward(next_blind, a2, none) == ANTE_BONUS * 2

    # joker term counts mult/xmult only, never chips (no double-count)
    fired = FakeScore(per_joker={"j_t": {"chips_added": 300, "mult_added": 5,
                                         "xmult_factor": 3.0}})
    r = reward(s0, half, fired)
    assert abs(r - (150 * SCALE_CHIPS + 5 * SCALE_MULT + math.log(3.0))) < 1e-9

    # terminals dominate and carry no shaping terms
    won = {"state": "GAME_OVER", "won": True}
    lost = {"state": "GAME_OVER", "won": False}
    assert reward(a2, won, None) == WIN_BONUS
    assert reward(a2, lost, None) == -LOSS_PENALTY

    # edge cases
    assert reward(None, {"round": {}}, None) == 0.0
    assert reward(None, won, None) == WIN_BONUS

    print("PASS")
