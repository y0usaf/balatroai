"""Baseline bot: legal-but-clueless. This is the bare core (doctrine 06) —
the runner + this bot must always complete a game with zero policy.
"""

from __future__ import annotations

import random

from . import Action, register


@register
class RandomBot:
    name = "random"

    def act(self, state: dict) -> Action:
        match state.get("state"):
            case "BLIND_SELECT":
                return Action("select", note="select blind")
            case "SELECTING_HAND":
                n = len(state.get("hand", {}).get("cards", []))
                k = random.randint(1, min(5, max(1, n)))
                cards = sorted(random.sample(range(n), k))
                return Action("play", {"cards": cards}, note=f"play {k} random")
            case "ROUND_EVAL":
                return Action("cash_out", note="cash out")
            case "SHOP":
                return Action("next_round", note="skip shop")
            case "SMODS_BOOSTER_OPENED":
                return Action("pack", {"skip": True}, note="skip pack")
            case _:
                return Action("gamestate", note="wait")
