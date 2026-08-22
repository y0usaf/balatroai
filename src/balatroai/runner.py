"""Game runner: drives one bot through one game.

Snapshot in, actions out (doctrine 02): the bot only ever sees the gamestate
dict and returns an Action; the runner owns all server communication, applies
per-state fallbacks when an action is rejected, and enforces a step watchdog.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from .bots import Action, Bot
from .client import Client, RPCError

# Safe action per state when the bot's action is rejected. Each fallback
# advances the game, so a rejected action can never loop forever.
FALLBACKS: dict[str, Action] = {
    "BLIND_SELECT": Action("select"),
    "SELECTING_HAND": Action("play", {"cards": [0]}),
    "ROUND_EVAL": Action("cash_out"),
    "SHOP": Action("next_round"),
    "SMODS_BOOSTER_OPENED": Action("pack", {"skip": True}),
}

# prev = state snapshot taken before the action was dispatched (for narration).
Emit = Callable[[dict, Action, str | None, dict | None], None]


@dataclass
class GameResult:
    won: bool
    ante: int
    round: int
    seed: str
    steps: int


class Runner:
    def __init__(self, client: Client, bot: Bot, emit: Emit | None = None,
                 max_steps: int = 3000):
        self.client = client
        self.bot = bot
        self.emit = emit or (lambda *_: None)
        self.max_steps = max_steps

    def play_game(self, deck: str = "RED", stake: str = "WHITE",
                  seed: str | None = None) -> GameResult:
        try:
            self.client.call("menu")
        except RPCError:
            pass  # already at menu
        params: dict = {"deck": deck, "stake": stake}
        if seed:
            params["seed"] = seed
        state = self.client.call("start", params)

        steps = 0
        strikes = 0
        while state.get("state") != "GAME_OVER" and steps < self.max_steps:
            steps += 1
            prev_state = state
            action = self.bot.act(state)
            error = None
            try:
                state = self.client.call(action.method, action.params)
                strikes = 0
            except RPCError as e:
                # Isolated failures are usually transient — either the game is
                # mid-animation (a cash-out still scoring while the snapshot
                # already says SHOP, so even vanilla can_use_consumeable()
                # says "not yet") or a reroll is refilling the shop UI.
                # Give it a beat, refresh the snapshot, and let the bot
                # replan instead of advancing past the phase. Third strike in
                # a row falls back to the phase action, which always advances,
                # so a persistently bad action can't loop.
                error = str(e)
                strikes += 1
                try:
                    if strikes >= 3:
                        fb = FALLBACKS.get(state.get("state", ""), Action("gamestate"))
                        state = self.client.call(fb.method, fb.params)
                    else:
                        time.sleep(1.0)
                        state = self.client.call("gamestate")
                except RPCError:
                    state = self.client.call("gamestate")
            self.emit(state, action, error, prev_state)

        return GameResult(
            won=bool(state.get("won")),
            ante=state.get("ante_num", 0),
            round=state.get("round_num", 0),
            seed=state.get("seed", seed or "?"),
            steps=steps,
        )
