"""Dense RL reward for balatroAI.

Additive terms, each a pure function of (prev_state, state, last_score,
action), composed by ``reward()``.  To tune or ablate, edit ``TERMS`` —
flip a weight to 0, adjust a scale, add a Term.  No other file needs to
change; the module stays side-effect free.

Terms (weights chosen so milestones dominate per-step shaping and
terminals dominate everything):

  score           fraction of the CURRENT blind cleared this step.
                  Scale-invariant: clearing a 300-chip ante-1 blind counts
                  the same as clearing a 30k-chip ante-8 blind, unlike the
                  old absolute chips/1000 whose value drifted 100x across
                  antes.  Capped at the target (no overkill farming) and
                  gated to within-one-blind steps (the chip counter resets
                  across transitions).  Falls back to absolute/1000 when no
                  blind target is visible.
  jokers          per-joker mult_added/10 + log(xmult_factor).  chips_added
                  already lands in score progress; counting it twice biased
                  toward chip jokers.
  money           money_delta * 0.01 — a means, kept small.
  ante            ANTE_BONUS * new_ante on ante-up — later antes are
                  harder, so their milestone is worth more.
  boss_violation  -2 when a played hand violates the CURRENT blind's
                  card-count rule (The Psychic demands >=5 cards).  The
                  selection layer obeys these constraints already, so this
                  fires only as a safety net — and as real signal once card
                  choice moves into the policy.
  terminal        +WIN_BONUS on a win, -LOSS_PENALTY on a loss; returned
                  alone, no shaping noise on the final step.

VecNormalize(norm_reward=True) absorbs residual scale drift.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from ..bots.heuristic import boss_card_limits

ANTE_BONUS = 5.0
WIN_BONUS = 50.0
LOSS_PENALTY = 20.0
SCALE_CHIPS = 1.0 / 1000.0  # legacy fallback when no blind target visible


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
    """Chips banked toward the current blind, capped at its target."""
    if not state:
        return 0.0
    chips = float(state.get("round", {}).get("chips", 0.0))
    target = blind_target(state)
    return min(chips, target) if target > 0 else chips


def _same_blind(prev: dict | None, state: dict) -> bool:
    return bool(prev) and (
        prev.get("ante_num"), prev.get("round_num"),
    ) == (state.get("ante_num"), state.get("round_num"))


def _score_progress(prev: dict | None, state: dict, last_score,
                    action=None) -> float:
    if not _same_blind(prev, state):
        return 0.0
    target = blind_target(state)
    if target <= 0:  # target invisible: legacy absolute shaping
        p = float(state.get("round", {}).get("chips", 0.0)) if state else 0.0
        q = float(prev.get("round", {}).get("chips", 0.0)) if prev else 0.0
        return (p - q) * SCALE_CHIPS
    return (_progress(state) - _progress(prev)) / target


def _jokers(prev: dict | None, state: dict, last_score, action=None) -> float:
    per_joker = getattr(last_score, "per_joker", None) or {}
    total = 0.0
    for e in per_joker.values():
        total += _f(e, "mult_added") * 0.1
        total += math.log(max(1.0, _f(e, "xmult_factor", 1.0)))
    return total


def _money(prev: dict | None, state: dict, last_score, action=None) -> float:
    m = float(state.get("money", 0.0)) - (
        float(prev.get("money", 0.0)) if prev else 0.0)
    return m * 0.01


def _ante(prev: dict | None, state: dict, last_score, action=None) -> float:
    now = state.get("ante_num", 0)
    was = prev.get("ante_num", 0) if prev else 0
    return ANTE_BONUS * now if now > was else 0.0


def _boss_violation(prev: dict | None, state: dict, last_score,
                    action=None) -> float:
    """1 when a played hand violates the CURRENT blind's card-count rule.
    Selection obeys these constraints already; this is the safety net."""
    if action is None or action.method != "play":
        return 0.0
    lo, hi = boss_card_limits(state)
    n = len((action.params or {}).get("cards") or [])
    return 1.0 if n and (n < lo or n > hi) else 0.0


@dataclass(frozen=True)
class Term:
    name: str
    weight: float
    fn: Callable[..., float]


TERMS: tuple[Term, ...] = (
    Term("score", 5.0, _score_progress),
    Term("jokers", 1.0, _jokers),
    Term("money", 1.0, _money),
    Term("ante", 1.0, _ante),
    Term("boss_violation", -2.0, _boss_violation),
)


def reward(prev_state: dict | None, state: dict, last_score,
           action=None) -> float:
    """Dense reward. last_score may be None (no hand yet); states optional."""
    if state.get("state") == "GAME_OVER":
        return WIN_BONUS if state.get("won") else -LOSS_PENALTY
    return sum(t.weight * t.fn(prev_state, state, last_score, action)
               for t in TERMS)


def reward_breakdown(prev_state: dict | None, state: dict, last_score,
                     action=None) -> dict[str, float]:
    """Per-term weighted contributions — for logs and debugging."""
    if state.get("state") == "GAME_OVER":
        return {"terminal": WIN_BONUS if state.get("won") else -LOSS_PENALTY}
    return {t.name: t.weight * t.fn(prev_state, state, last_score, action)
            for t in TERMS}


if __name__ == "__main__":
    from collections import namedtuple

    FakeScore = namedtuple("FakeScore", "per_joker chips mult total",
                           defaults=({}, 0, 0, 0))
    none = FakeScore(per_joker={})
    blind1 = {"small": {"status": "CURRENT", "score": 300}}

    # no change -> 0
    s0 = {"round": {"chips": 0.0}, "ante_num": 1, "round_num": 1,
          "blinds": blind1, "money": 4}
    assert abs(reward(s0, dict(s0), none)) < 1e-9

    # score progress is fractional and capped at the target
    half = {**s0, "round": {"chips": 150.0}}
    sealed = {**s0, "round": {"chips": 900.0}}
    assert abs(reward(s0, half, none) - 5.0 * 0.5) < 1e-9          # half blind
    assert abs(reward(half, sealed, none) - 5.0 * 0.5) < 1e-9      # capped

    # scale-invariance: same fractions at a 30k-chip ante-8 blind
    big = {"small": {"status": "CURRENT", "score": 30000}}
    b0 = {"round": {"chips": 0.0}, "ante_num": 8, "round_num": 22,
          "blinds": big, "money": 4}
    bh = {**b0, "round": {"chips": 15000.0}}
    assert abs(reward(b0, bh, none) - 5.0 * 0.5) < 1e-9            # identical

    # blind transition resets the counter without a negative spike
    nxt = {"round": {"chips": 0.0}, "ante_num": 1, "round_num": 2,
           "blinds": {"big": {"status": "CURRENT", "score": 600}},
           "money": 4}
    assert reward(sealed, nxt, none) == 0.0

    # no target visible: legacy absolute shaping (still weighted by TERM)
    raw_prev = {"round": {"chips": 0.0}, "ante_num": 1, "round_num": 1}
    raw = {"round": {"chips": 500.0}, "ante_num": 1, "round_num": 1}
    assert abs(reward(raw_prev, raw, none) - 5.0 * 0.5) < 1e-9

    # ante milestone scales with the ante reached
    a2 = {**nxt, "ante_num": 2, "round_num": 3}
    got = reward(nxt, a2, none)
    assert abs(got - ANTE_BONUS * 2) < 1e-9, got

    # joker term counts mult/xmult only, never chips
    fired = FakeScore(per_joker={"j_t": {"chips_added": 300, "mult_added": 5,
                                         "xmult_factor": 3.0}})
    r = reward(s0, half, fired)
    assert abs(r - (2.5 + 0.5 + math.log(3.0))) < 1e-9, r

    # boss violation: Psychic active, playing 3 cards costs -2
    psychic = {"small": {"status": "CURRENT", "name": "The Psychic",
                         "score": 300}}
    ps0 = {"round": {"chips": 0.0}, "ante_num": 1, "round_num": 1,
           "blinds": psychic, "money": 4}
    ps_half = {**ps0, "round": {"chips": 150.0}}

    class Play3:
        method = "play"
        params = {"cards": [0, 1, 2]}

    class Play5:
        method = "play"
        params = {"cards": [0, 1, 2, 3, 4]}

    class Toss4:
        method = "discard"
        params = {"cards": [0, 1, 2, 3]}

    bd = reward_breakdown(ps0, ps_half, none, Play3())
    assert bd["boss_violation"] == -2.0, bd
    assert reward_breakdown(ps0, ps_half, none, Play5())["boss_violation"] == 0.0
    assert reward_breakdown(ps0, ps_half, none, Toss4())["boss_violation"] == 0.0
    assert reward_breakdown(half, sealed, none)["boss_violation"] == 0.0  # no action

    # terminals dominate and carry no shaping terms
    won = {"state": "GAME_OVER", "won": True}
    lost = {"state": "GAME_OVER", "won": False}
    assert reward(a2, won, None) == WIN_BONUS
    assert reward(a2, lost, None) == -LOSS_PENALTY

    # breakdown mirrors reward() and names its terms
    bd = reward_breakdown(s0, half, fired)
    assert set(bd) == {"score", "jokers", "money", "ante", "boss_violation"}
    assert abs(sum(bd.values()) - r) < 1e-9

    # edge cases
    assert reward(None, {"round": {}}, None) == 0.0
    assert reward(None, won, None) == WIN_BONUS

    print("PASS")
