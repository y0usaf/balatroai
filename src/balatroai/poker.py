"""Poker hand evaluation for Balatro. Stdlib only, runnable standalone.

`python -m balatroai.poker` runs the self-test (used by the bare-core CI check).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations

RANK_VAL = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}
CHIP_VAL = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "T": 10, "J": 10, "Q": 10, "K": 10, "A": 11,
}

# Level-1 base (chips, mult) — fallback when live values are unavailable.
BASE_HANDS: dict[str, tuple[int, int]] = {
    "High Card": (5, 1),
    "Pair": (10, 2),
    "Two Pair": (20, 2),
    "Three of a Kind": (30, 3),
    "Straight": (30, 4),
    "Flush": (35, 4),
    "Full House": (40, 4),
    "Four of a Kind": (60, 7),
    "Straight Flush": (100, 8),
    "Five of a Kind": (120, 12),
    "Flush House": (140, 14),
    "Flush Five": (160, 16),
}


@dataclass
class PlayChoice:
    indices: list[int]          # hand indices to play (server order)
    name: str                   # poker hand name
    score: int                  # estimated (chips * mult)
    scoring: list[int] = field(default_factory=list)  # subset of indices that score


def _is_straight(vals: list[int]) -> bool:
    s = sorted(set(vals))
    if len(s) != 5:
        return False
    return s[-1] - s[0] == 4 or s == [2, 3, 4, 5, 14]


def classify(ranks: list[str], suits: list[str]) -> tuple[str, list[int]]:
    """Classify a played subset. Returns (hand name, positions that score)."""
    n = len(ranks)
    counts = Counter(ranks)
    groups = sorted(counts.values(), reverse=True)
    flush = n == 5 and len(set(suits)) == 1
    straight = n == 5 and _is_straight([RANK_VAL[r] for r in ranks])
    all_pos = list(range(n))

    if groups[0] == 5:
        return ("Flush Five" if flush else "Five of a Kind"), all_pos
    if flush and groups[:2] == [3, 2]:
        return "Flush House", all_pos
    if flush and straight:
        return "Straight Flush", all_pos
    if groups[0] == 4:
        quad = next(r for r, c in counts.items() if c == 4)
        return "Four of a Kind", [i for i, r in enumerate(ranks) if r == quad]
    if groups[:2] == [3, 2]:
        return "Full House", all_pos
    if flush:
        return "Flush", all_pos
    if straight:
        return "Straight", all_pos
    if groups[0] == 3:
        trip = next(r for r, c in counts.items() if c == 3)
        return "Three of a Kind", [i for i, r in enumerate(ranks) if r == trip]
    if groups[:2] == [2, 2]:
        pairs = [r for r, c in counts.items() if c == 2]
        return "Two Pair", [i for i, r in enumerate(ranks) if r in pairs]
    if groups[0] == 2:
        pr = next(r for r, c in counts.items() if c == 2)
        return "Pair", [i for i, r in enumerate(ranks) if r == pr]
    hi = max(range(n), key=lambda i: RANK_VAL[ranks[i]])
    return "High Card", [hi]


def hand_value(name: str, hands_info: dict | None) -> tuple[int, int]:
    """(chips, mult) for a hand type, live values if present, else base."""
    if hands_info and name in hands_info:
        h = hands_info[name]
        chips, mult = h.get("chips", 0), h.get("mult", 0)
        if chips and mult:
            return chips, mult
    return BASE_HANDS[name]


def best_play(cards: list[dict], hands_info: dict | None = None) -> PlayChoice | None:
    """Best subset (size 1..5) of a balatrobot hand-card list, by estimated score.

    Cards without a rank (e.g. Stone cards) are ignored for hand-building.
    """
    ranked = [
        (i, c["value"]["rank"], c["value"].get("suit", "?"))
        for i, c in enumerate(cards)
        if c.get("value", {}).get("rank")
    ]
    if not ranked:
        return None

    best: PlayChoice | None = None
    for size in range(1, min(5, len(ranked)) + 1):
        for combo in combinations(ranked, size):
            ranks = [r for _, r, _ in combo]
            suits = [s for _, _, s in combo]
            name, scoring_pos = classify(ranks, suits)
            chips, mult = hand_value(name, hands_info)
            score = (chips + sum(CHIP_VAL[ranks[p]] for p in scoring_pos)) * mult
            if best is None or score > best.score or (
                score == best.score and size < len(best.indices)
            ):
                best = PlayChoice(
                    indices=[i for i, _, _ in combo],
                    name=name,
                    score=score,
                    scoring=[combo[p][0] for p in scoring_pos],
                )
    return best


def discard_candidates(cards: list[dict], keep: set[int], limit: int = 5) -> list[int]:
    """Indices to discard: everything not in `keep`, lowest rank first, capped."""
    idx = [
        i for i, c in enumerate(cards)
        if i not in keep and c.get("value", {}).get("rank")
    ]
    idx.sort(key=lambda i: RANK_VAL[cards[i]["value"]["rank"]])
    return idx[:limit]


def keep_set(cards: list[dict]) -> set[int]:
    """Cards worth keeping when fishing: 4+ flush draw > paired ranks > top 2 ranks."""
    ranked = [
        (i, c["value"]["rank"], c["value"].get("suit", "?"))
        for i, c in enumerate(cards)
        if c.get("value", {}).get("rank")
    ]
    suits = Counter(s for _, _, s in ranked)
    if suits and suits.most_common(1)[0][1] >= 4:
        suit = suits.most_common(1)[0][0]
        return {i for i, _, s in ranked if s == suit}
    ranks = Counter(r for _, r, _ in ranked)
    paired = {r for r, c in ranks.items() if c >= 2}
    if paired:
        return {i for i, r, _ in ranked if r in paired}
    top = sorted(ranked, key=lambda t: RANK_VAL[t[1]], reverse=True)[:2]
    return {i for i, _, _ in top}


def _selftest() -> None:
    def cards(*rs: str) -> list[dict]:
        return [{"value": {"rank": r[0], "suit": r[1]}} for r in rs]

    assert classify(["K", "K"], ["S", "D"]) == ("Pair", [0, 1])
    assert classify(["A", "2", "3", "4", "5"], list("SDCHS"))[0] == "Straight"
    assert classify(["9", "T", "J", "Q", "K"], list("SSSSS"))[0] == "Straight Flush"
    assert classify(["7", "7", "7", "9", "9"], list("SDCHS"))[0] == "Full House"
    assert classify(["A", "A", "A", "A", "A"], list("SSSSS"))[0] == "Flush Five"

    b = best_play(cards("KS", "KD", "2C", "7H", "9S", "3D", "4D", "8C"))
    assert b and b.name == "Pair" and len(b.indices) == 2

    b = best_play(cards("2S", "3S", "4S", "5S", "6S", "KD", "KC", "9H"))
    assert b and b.name == "Straight Flush"

    ks = keep_set(cards("2H", "5H", "9H", "KH", "3C", "7D", "8S", "TC"))
    assert ks == {0, 1, 2, 3}  # flush draw in hearts

    d = discard_candidates(cards("KS", "KD", "2C", "7H", "9S", "3D", "4D", "8C"), {0, 1})
    assert d == [2, 5, 6, 3, 7]  # lowest first, capped at 5

    print("poker selftest: ok")


if __name__ == "__main__":
    _selftest()
