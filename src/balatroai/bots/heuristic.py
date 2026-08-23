"""Heuristic bot: plays the best poker hand, fishes with discards when the
current hand is weak, uses planets, buys jokers/planets in the shop.
Watchable and non-embarrassing; not remotely optimal.
"""

from __future__ import annotations

from .. import poker
from . import Action, register

WEAK_HANDS = {"High Card", "Pair", "Two Pair"}

# Jokers worth selling: objectively weak (+4 mult) or money-draining/expiring.
WEAK_JOKERS = {"j_joker"}


# Boss blinds constraining how many cards a played hand may contain,
# keyed by blind name as reported in the gamestate (lowercased).
_MIN_CARDS_BY_BOSS = {"the psychic": 5}
_MAX_CARDS_BY_BOSS: dict[str, int] = {}  # none in vanilla
_RANK_ORDER = "23456789TJQKA"


def boss_card_limits(state: dict) -> tuple[int, int]:
    """(min, max) cards a played hand may have under the CURRENT blind."""
    blind = current_blind(state)
    if not blind:
        return 0, 5
    name = str(blind.get("name", "")).lower()
    return (_MIN_CARDS_BY_BOSS.get(name, 0),
            min(_MAX_CARDS_BY_BOSS.get(name, 5), 5))


def current_blind(state: dict) -> dict | None:
    for blind in (state.get("blinds") or {}).values():
        if isinstance(blind, dict) and blind.get("status") == "CURRENT":
            return blind
    return None


@register
class HeuristicBot:
    name = "heuristic"

    # Optional ContributionLedger injected by the driving env; when present
    # and fully observed, _sell evicts the empirically weakest holder.
    contribution = None

    # keep this much money in the shop from ante 3 on (interest-ish reserve)
    reserve_from_ante = 3
    reserve = 10

    def act(self, state: dict) -> Action:
        match state.get("state"):
            case "BLIND_SELECT":
                blind = next(
                    (b for b in (state.get("blinds") or {}).values()
                     if isinstance(b, dict) and b.get("status") == "SELECT"),
                    None,
                )
                name = blind.get("name", "blind") if blind else "blind"
                return Action("select", note=f"take on {name}")
            case "SELECTING_HAND":
                return self._hand(state)
            case "ROUND_EVAL":
                return Action("cash_out", note="cash out")
            case "SHOP":
                return self._shop(state)
            case "SMODS_BOOSTER_OPENED":
                return self._pack(state)
            case _:
                return Action("gamestate", note="wait")

    def _hand(self, state: dict) -> Action:
        cards = state.get("hand", {}).get("cards", [])
        choice = poker.best_play(cards, state.get("hands"))
        if choice is None:
            idx = self._pad(cards, [0] if cards else [], 1)
            return Action("play", {"cards": idx},
                          note="play first (no ranked cards)")

        rnd = state.get("round", {})
        hands_left = rnd.get("hands_left", 1)
        discards_left = rnd.get("discards_left", 0)
        blind = current_blind(state)
        needed = None
        if blind:
            needed = max(0, blind.get("score", 0) - rnd.get("chips", 0))

        # Boss card-count constraints, plus Splash: it pays +4 mult for
        # every played card whether scored or not, so pad to a full hand.
        lo, hi = boss_card_limits(state)
        want = max(lo, 5 if self._splash_owned(state) else 0)
        idx = list(choice.indices)
        fillers = 0
        if len(idx) < want:
            padded = self._pad(cards, idx, min(want, hi))
            fillers = len(padded) - len(idx)
            idx = padded

        label = f"{choice.name} ~{choice.score}"
        if fillers:
            label += f" +{fillers}"
        if needed is not None and choice.score >= needed:
            return Action("play", {"cards": idx}, note=f"play {label} (seals it)")

        weak = choice.name in WEAK_HANDS
        behind = needed is not None and hands_left > 0 and choice.score * hands_left < needed
        if discards_left > 0 and weak and (behind or needed is None) and not fillers:
            keep = poker.keep_set(cards)
            toss = poker.discard_candidates(cards, keep)
            if toss:
                return Action("discard", {"cards": toss}, note=f"fish: toss {len(toss)}")
        return Action("play", {"cards": idx}, note=f"play {label}")

    @staticmethod
    def _splash_owned(state: dict) -> bool:
        return any(j.get("key") == "j_splash"
                   for j in (state.get("jokers") or {}).get("cards", []))

    @staticmethod
    def _pad(cards: list[dict], indices: list[int], target: int) -> list[int]:
        """Extend a played-hand index list with the strongest unused cards."""
        used = set(indices)
        rest = sorted(
            (i for i in range(len(cards)) if i not in used),
            key=lambda i: _RANK_ORDER.index(
                ((cards[i].get("value") or {}).get("rank") or "2")[0]),
            reverse=True,
        )
        out = list(indices)
        while len(out) < target and rest:
            out.append(rest.pop(0))
        return out
    def _shop(self, state: dict) -> Action:
        money = state.get("money", 0)
        ante = state.get("ante_num", 1)
        reserve = self.reserve if ante >= self.reserve_from_ante else 0

        # use held planets first
        held = state.get("consumables", {}).get("cards", [])
        for i, c in enumerate(held):
            if c.get("set") == "PLANET":
                return Action("use", {"consumable": i}, note=f"use {c.get('label')}")

        jokers = state.get("jokers", {})
        joker_room = jokers.get("count", 0) < jokers.get("limit", 5)
        shop = state.get("shop", {}).get("cards", [])
        for i, c in enumerate(shop):
            cost = c.get("cost", {}).get("buy", 999)
            if money - cost < reserve:
                continue
            if c.get("set") == "JOKER" and joker_room:
                return Action("buy", {"card": i}, note=f"buy {c.get('label')} (${cost})")
            if c.get("set") == "PLANET":
                return Action("buy", {"card": i}, note=f"buy {c.get('label')} (${cost})")

        # Rack full but a better joker is on sale: evict the measured dud
        # (ledger) or a known-weak holder to make room for it.
        if not joker_room:
            upgrade = next((c for c in shop
                            if c.get("set") == "JOKER"
                            and money - c.get("cost", {}).get("buy", 999) >= reserve),
                           None)
            dud = self._sell(state) or self._cheapest_holder(state)
            if upgrade is not None and dud is not None:
                return Action(dud.method, dud.params,
                              note=f"{dud.note} -> make room for {upgrade.get('label')}")
        return Action("next_round", note="leave shop")

    # -- selling / rerolling ---------------------------------------------
    def _sell(self, state: dict) -> Action | None:
        """Sell the weakest held joker: measured dud first, then the
        rental/perishable/known-weak list."""
        jokers = state.get("jokers", {})
        cards = jokers.get("cards", [])
        ledger = self.contribution
        if ledger is not None:
            weak_key = ledger.weakest_key(state)
            if weak_key is not None:
                for i, j in enumerate(cards):
                    if j.get("key") == weak_key:
                        return Action("sell", {"joker": i},
                                      note=f"sell {j.get('label')} (ledger)")
        for i, j in enumerate(cards):
            mod = j.get("modifier", {}) or {}
            if mod.get("rental") or mod.get("perishable") or j.get("key") in WEAK_JOKERS:
                return Action("sell", {"joker": i}, note=f"sell {j.get('label')}")
        return None

    def _cheapest_holder(self, state: dict) -> Action | None:
        """Sell-cost fallback: evict the least invested non-eternal holder."""
        best, best_cost = None, None
        for i, j in enumerate((state.get("jokers") or {}).get("cards", [])):
            if (j.get("modifier") or {}).get("eternal"):
                continue
            cost = (j.get("cost") or {}).get("sell", 999)
            if best_cost is None or cost < best_cost:
                best, best_cost = i, cost
        if best is None:
            return None
        card = ((state.get("jokers") or {}).get("cards") or [])[best]
        return Action("sell", {"joker": best},
                      note=f"sell {card.get('label', '?')} (cheapest)")

    def _reroll(self, state: dict) -> Action | None:
        """Reroll the shop if we can afford it above reserve, else None."""
        money = state.get("money", 0)
        ante = state.get("ante_num", 1)
        reserve = self.reserve if ante >= self.reserve_from_ante else 0
        reroll_cost = state.get("round", {}).get("reroll_cost", 5)
        if money - reroll_cost >= reserve:
            return Action("reroll", note=f"reroll (${reroll_cost})")
        return None

    # -- packs ------------------------------------------------------------
    def _pack(self, state: dict) -> Action:
        cards = state.get("pack", {}).get("cards", [])
        jokers = state.get("jokers", {})
        joker_room = jokers.get("count", 0) < jokers.get("limit", 5)
        for i, c in enumerate(cards):
            if c.get("set") == "JOKER" and joker_room:
                return Action("pack", {"card": i}, note=f"take {c.get('label')}")
        for i, c in enumerate(cards):
            if c.get("set") == "PLANET":
                return Action("pack", {"card": i}, note=f"take {c.get('label')}")
        for i, c in enumerate(cards):
            if c.get("set") == "DEFAULT":
                return Action("pack", {"card": i}, note=f"take {c.get('label')}")
        return Action("pack", {"skip": True}, note="skip pack")
