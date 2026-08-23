"""Per-joker contribution ledger.

Accumulates each held joker's realized output across scored hands so
decisions — sell the dud, keep the engine — can key off measured value
instead of a hardcoded weak-list.  Pure bookkeeping: no I/O, no game
knowledge beyond the score-entry shape.
"""

from __future__ import annotations

import math

# Scales putting chips/mult/xmult on one comparable axis.  Arbitrary by
# design — this is a ranking signal, not an exact price.
_CHIPS = 1.0 / 100.0
_MULT = 1.0
_XMULT = 25.0


def hand_value(entry: dict) -> float:
    """One scoring-hand contribution, normalized to a single axis."""
    return (float(entry.get("chips_added", 0) or 0) * _CHIPS
            + float(entry.get("mult_added", 0) or 0) * _MULT
            + math.log(max(1.0, float(entry.get("xmult_factor", 1.0) or 1)))
            * _XMULT)


class ContributionLedger:
    """joker key -> cumulative normalized earnings while held."""

    def __init__(self) -> None:
        self.earned: dict[str, float] = {}

    def observe(self, last_score) -> None:
        """Fold one scoring hand's per-joker entries into the ledger."""
        for key, entry in (getattr(last_score, "per_joker", None) or {}).items():
            self.earned[key] = self.earned.get(key, 0.0) + hand_value(entry)

    def values_for(self, state: dict | None) -> list[float]:
        """Ledger value aligned with state's held-joker order."""
        held = ((state or {}).get("jokers") or {}).get("cards") or []
        return [self.earned.get((j or {}).get("key") or "", 0.0)
                for j in held]

    def weakest_key(self, state: dict) -> str | None:
        """Held key with the lowest measured contribution.

        Returns None unless every incumbent has at least one observed
        scoring hand — an unobserved joker is unknown, not weak.
        """
        cards = (state.get("jokers") or {}).get("cards", [])
        keys = [(j or {}).get("key") or "" for j in cards]
        if not keys or any(k not in self.earned for k in keys):
            return None
        return min(keys, key=lambda k: self.earned[k])


if __name__ == "__main__":
    from collections import namedtuple

    Score = namedtuple("Score", "per_joker")
    st = {"jokers": {"cards": [{"key": "j_a"}, {"key": "j_b"}]}}
    led = ContributionLedger()

    # unobserved -> no weakest verdict
    assert led.weakest_key(st) is None

    # observations accumulate per key on a single comparable axis
    led.observe(Score(per_joker={"j_a": {"chips_added": 300, "mult_added": 2,
                                         "xmult_factor": 1.0}}))
    led.observe(Score(per_joker={"j_b": {"chips_added": 10, "mult_added": 40,
                                         "xmult_factor": 1.5}}))
    va, vb = led.values_for(st)
    assert abs(va - (300 / 100 + 2)) < 1e-9
    assert abs(vb - (10 / 100 + 40 + 25 * math.log(1.5))) < 1e-9

    # repeated observation identifies the clear dud
    for _ in range(5):
        led.observe(Score(per_joker={"j_a": {"chips_added": 0,
                                             "mult_added": 1,
                                             "xmult_factor": 1.0}}))
        led.observe(Score(per_joker={"j_b": {"chips_added": 400,
                                             "mult_added": 20,
                                             "xmult_factor": 3.0}}))
    assert led.weakest_key(st) == "j_a"
    assert led.values_for(st)[1] > 50

    # empty holders -> None
    assert led.weakest_key({"jokers": {"cards": []}}) is None

    print("PASS ledger")
