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


def current_blind(state: dict) -> dict | None:
    for blind in (state.get("blinds") or {}).values():
        if isinstance(blind, dict) and blind.get("status") == "CURRENT":
            return blind
    return None


@register
class HeuristicBot:
    name = "heuristic"

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

    # -- playing ----------------------------------------------------------
    def _hand(self, state: dict) -> Action:
        cards = state.get("hand", {}).get("cards", [])
        choice = poker.best_play(cards, state.get("hands"))
        if choice is None:
            return Action("play", {"cards": [0]}, note="play first (no ranked cards)")

        rnd = state.get("round", {})
        hands_left = rnd.get("hands_left", 1)
        discards_left = rnd.get("discards_left", 0)
        blind = current_blind(state)
        needed = None
        if blind:
            needed = max(0, blind.get("score", 0) - rnd.get("chips", 0))

        label = f"{choice.name} ~{choice.score}"
        if needed is not None and choice.score >= needed:
            return Action("play", {"cards": choice.indices}, note=f"play {label} (seals it)")

        weak = choice.name in WEAK_HANDS
        behind = needed is not None and hands_left > 0 and choice.score * hands_left < needed
        if discards_left > 0 and weak and (behind or needed is None):
            keep = poker.keep_set(cards)
            toss = poker.discard_candidates(cards, keep)
            if toss:
                return Action("discard", {"cards": toss}, note=f"fish: toss {len(toss)}")
        return Action("play", {"cards": choice.indices}, note=f"play {label}")

    # -- shopping ---------------------------------------------------------
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
        return Action("next_round", note="leave shop")

    # -- selling / rerolling ---------------------------------------------
    def _sell(self, state: dict) -> Action | None:
        """Sell the weakest held joker (rental/perishable/known-weak), else None."""
        jokers = state.get("jokers", {})
        for i, j in enumerate(jokers.get("cards", [])):
            mod = j.get("modifier", {}) or {}
            if mod.get("rental") or mod.get("perishable") or j.get("key") in WEAK_JOKERS:
                return Action("sell", {"joker": i}, note=f"sell {j.get('label')}")
        return None

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
