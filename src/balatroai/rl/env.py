"""Gymnasium RL environment over the jackdaw sim (first cut).

``BalatroEnv`` wraps a :class:`balatroai.sim.SimClient` in a
``gymnasium.Env``.  The action space is a coarse intent set; the concrete
card-selection details are delegated to :class:`HeuristicBot` (its
``_hand`` / ``_shop`` / ``_pack`` helpers) so the agent only has to pick a
high-level strategy and the sim always gets a legal, well-selected action.

This module intentionally does NOT import torch and does NOT pull in any
training stack.  Run the self-check with the ephemeral gymnasium only:

    uv sync --extra sim
    uv run --with gymnasium python src/balatroai/rl/env.py
"""

from __future__ import annotations

try:  # package/relative imports (normal usage)
    from .reward import reward
    from ..bots import Action
    from ..bots.heuristic import HeuristicBot
    from ..runner import FALLBACKS, RPCError
    from ..sim import SimClient
except ImportError:  # run as a bare script (python src/.../rl/env.py)
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from balatroai.rl.reward import reward
    from balatroai.bots import Action
    from balatroai.bots.heuristic import HeuristicBot
    from balatroai.runner import FALLBACKS, RPCError
    from balatroai.sim import SimClient

# gymnasium is optional for the env itself; the SB3 training stack ("train"
# extra) installs it.  When present we subclass the real ``gymnasium.Env`` so
# stable-baselines3 can wrap us directly; otherwise we fall back to a plain
# object with the minimal stdlib shims below, keeping the core stdlib-only.
try:
    import gymnasium as gym
except Exception:
    gym = None

_EnvBase = gym.Env if gym is not None else object

# ---------------------------------------------------------------------------
# Card / obs layout constants
# ---------------------------------------------------------------------------

# Hand slots to pad/truncate a hand into a fixed-size vector.
HAND_SLOTS = 8

# Rank letter -> scalar (0 = empty slot, 1..13 for 2..A).
_RANK_INDEX: dict[str, int] = {
    "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "9": 8,
    "T": 9, "J": 10, "Q": 11, "K": 12, "A": 13,
}
# Suit one-hot order (matches serialize_card suit letters).
_SUITS = ["S", "H", "D", "C"]

# Hardcoded bag of ~40 common joker center-keys (presence one-hot).  An extra
# trailing "other/unknown" slot absorbs any joker not in this list so rare or
# future jokers are still (coarsely) visible.
JOKER_KEYS: list[str] = [
    "j_joker", "j_greedy_joker", "j_lusty_joker", "j_wrathful_joker",
    "j_gluttenous_joker", "j_jolly", "j_zany", "j_mad", "j_crazy",
    "j_delayed_grat", "j_even_steven", "j_odd_todd", "j_scholar",
    "j_business", "j_supernova", "j_ride_the_bus", "j_photograph",
    "j_vagabond", "j_baron", "j_steel_joker", "j_fibonacci", "j_trousers",
    "j_walkie_talkie", "j_dusk", "j_raised_fist", "j_abstract", "j_banner",
    "j_blue_joker", "j_bull", "j_bootstraps", "j_sly", "j_clever", "j_devious",
    "j_crafty", "j_wily", "j_green_joker", "j_constellation", "j_8_ball",
    "j_dna", "j_hologram",
]
JOKER_KEYS_SET = frozenset(JOKER_KEYS)
N_SCALARS = 11
HAND_DIM = HAND_SLOTS * (_RANK_LEN := 1 + len(_SUITS))  # 8 * 5
JOKER_DIM = len(JOKER_KEYS) + 1  # +1 "other" slot
OBS_DIM = N_SCALARS + HAND_DIM + JOKER_DIM


# ---------------------------------------------------------------------------
# Obs layout (documentation)
# ---------------------------------------------------------------------------
_OBS_DOC = f"""\
Fixed-size observation vector, shape ({OBS_DIM},).

Block 1 - scalars ({N_SCALARS}):
  0  money
  1  ante_num
  2  round_num
  3  hands_left
  4  discards_left
  5  chips
  6  blind_score   (score needed to beat the CURRENT blind; 0 if none)
  7  hand_size     (hand.count)
  8  joker_count
  9  consumable_count
  10 shop_count

Block 2 - hand ({HAND_SLOTS} slots x {_RANK_LEN} = {HAND_DIM}):
  For each of the {HAND_SLOTS} hand slots (in order, left to right):
    [rank(0-13), suit_S, suit_H, suit_D, suit_C]
  rank: 0 = empty slot, 1..13 = 2..A.
  suit: one-hot; empty slot has all-zero suit bits.

Block 3 - jokers bag-of-keys ({len(JOKER_KEYS)}+1 = {JOKER_DIM}):
  One-hot presence of each of the first {len(JOKER_KEYS)} hardcoded joker
  keys in ``JOKER_KEYS``, followed by a final "other/unknown" bit set when a
  held joker key is not in the list.  Presence (not multiplicity).
"""


# Map raw intent -> logical action name (used for notes / validation).
_INTENT_NAMES = [
    "select", "play", "discard", "buy", "use",
    "next_round", "cash_out", "pack", "sell", "reroll",
]


class BalatroEnv(_EnvBase):
    """gymnasium.Env with a two-stage intent -> Action action space.

    Action space: ``spaces.Discrete(8)`` intents (see module docstring).

    No gymnasium subclass is required for the first cut; ``BalatroEnv``
    exposes the standard ``reset`` / ``step`` / ``obs`` API so it can be
    wrapped by ``gymnasium.wrappers.NormalizeObservation`` etc.
    """

    def __init__(self, deck: str = "RED", stake: str = "WHITE",
                 seed: str | None = None, max_steps: int = 3000):
        self.deck = deck
        self.stake = stake
        self.seed = seed
        self.max_steps = max_steps
        super().__init__()
        self.client = SimClient()
        self._bot = HeuristicBot()

        self.action_space = self._spaces_discrete(10)
        self.observation_space = self._observations_box()

        self.state: dict | None = None
        self._step_count = 0

    # -- minimal space shims (gymnasium optional; avoids hard import) -----
    def _spaces_discrete(self, n: int):
        try:
            import gymnasium.spaces as sp
            return sp.Discrete(n)
        except Exception:
            return _Discrete(n)

    def _observations_box(self):
        low = [0.0] * OBS_DIM
        high = [1e9] * OBS_DIM
        # rank field (slot-th slot) lives in [0, 13]
        for s in range(HAND_SLOTS):
            high[N_SCALARS + s * _RANK_LEN] = 13.0
            for k in range(1, _RANK_LEN):
                low[N_SCALARS + s * _RANK_LEN + k] = 0.0
                high[N_SCALARS + s * _RANK_LEN + k] = 1.0
        # joker presence bits in [0, 1]
        for i in range(N_SCALARS + HAND_DIM, OBS_DIM):
            high[i] = 1.0
        try:
            import gymnasium.spaces as sp
            return sp.Box(low=_lf(low), high=_lf(high), shape=(OBS_DIM,),
                          dtype="float32")
        except Exception:
            return _Box(_lf(low), _lf(high))

    # -- public API --------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        """(obs, info).  Start a fresh game via SimClient.call("start", ...)."""
        options = options or {}
        deck = options.get("deck", self.deck)
        stake = options.get("stake", self.stake)
        seed = options.get("seed", self.seed if self.seed else seed)
        # jackdaw wants an 8-char string seed; SB3 passes an int (or None).
        # Map int -> deterministic string so SB3's seed still controls
        # reproducibility, and None -> random string so training varies games.
        if seed is None:
            import random
            import string
            seed = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
        elif isinstance(seed, int):
            seed = f"{seed % 10**8:08d}"
        params: dict = {"deck": deck, "stake": stake, "seed": seed}
        try:
            self.client.call("menu")
        except RPCError:
            pass  # sim already at menu
        self.state = self.client.call("start", params)
        self._step_count = 0
        return self.obs(self.state), {"seed": self.state.get("seed"),
                                      "deck": deck, "stake": stake,
                                      "state": self.state}

    def step(self, action: int):
        """(obs, reward, terminated, truncated, info)."""
        if self.state is None:
            raise RuntimeError("reset() must be called before step()")
        intent = int(action)
        act = self._resolve_action(intent, self.state)
        prev_state = self.state
        try:
            self.state = self.client.call(act.method, act.params)
        except RPCError:
            fb = FALLBACKS.get(self.state.get("state", ""), Action("gamestate"))
            try:
                self.state = self.client.call(fb.method, fb.params)
            except RPCError:
                self.state = self.client.call("gamestate")
        last_score = self.client.last_score()
        r = reward(prev_state, self.state, last_score)
        terminated = bool(self.state.get("state") == "GAME_OVER")
        truncated = bool(self._step_count >= self.max_steps)
        self._step_count += 1
        info = {"state": self.state, "last_score": last_score,
                "action_method": act.method}
        return self.obs(self.state), float(r), terminated, truncated, info

    # -- intent -> concrete Action ----------------------------------------
    def _resolve_action(self, intent: int, state: dict) -> Action:
        return resolve_intent(int(intent), state, self._bot)

    # -- observation -------------------------------------------------------
    def obs(self, state: dict | None = None):
        """Feature vector for ``state`` (defaults to current)."""
        state = state if state is not None else self.state
        return encode_obs(state)

    @staticmethod
    def _blind_score(state: dict) -> float:
        return _blind_score(state)

# ---------------------------------------------------------------------------
# Reusable, SimClient-free pieces
# ---------------------------------------------------------------------------
# The module-level functions below contain the actual obs-encoding and
# intent->Action resolution logic.  ``BalatroEnv`` delegates to them, and a
# bot (e.g. PolicyBot) can reuse them without owning a SimClient.


def _blind_score(state: dict) -> float:
    """Score needed to beat the CURRENT blind (0 if none)."""
    for blind in (state.get("blinds") or {}).values():
        if isinstance(blind, dict) and blind.get("status") == "CURRENT":
            return float(blind.get("score", 0))
    return 0.0


def encode_obs(state: dict) -> list[float]:
    """Full fixed-size observation vector for ``state`` (see ``_OBS_DOC``).

    Pure: reads the gamestate snapshot, writes nothing; does not require a
    SimClient or a running game.  Returns a ``list[float]`` so downstream
    code (numpy etc.) can cast it as needed (matches ``_as_f32`` output).
    """
    rnd = state.get("round") or {}
    hand = state.get("hand") or {}
    jokers = state.get("jokers") or {}
    consumables = state.get("consumables") or {}
    shop = state.get("shop") or {}

    vec = [
        float(state.get("money", 0)),
        float(state.get("ante_num", 1)),
        float(state.get("round_num", 0)),
        float(rnd.get("hands_left", 0)),
        float(rnd.get("discards_left", 0)),
        float(rnd.get("chips", 0)),
        float(_blind_score(state)),
        float(hand.get("count", 0)),
        float(jokers.get("count", 0)),
        float(consumables.get("count", 0)),
        float(shop.get("count", 0)),
    ]

    hand_cards = hand.get("cards", [])
    for slot in range(HAND_SLOTS):
        if slot < len(hand_cards):
            val = (hand_cards[slot].get("value") or {})
            rank = _RANK_INDEX.get(val.get("rank", ""), 0)
            suit = val.get("suit", "")
        else:
            rank, suit = 0, ""
        vec.append(float(rank))
        for s in _SUITS:
            vec.append(1.0 if suit == s else 0.0)

    present = [0.0] * len(JOKER_KEYS)
    other = 0.0
    for card in jokers.get("cards", []):
        k = card.get("key")
        if k in JOKER_KEYS_SET:
            present[JOKER_KEYS.index(k)] = 1.0  # presence (bag)
        else:
            other = 1.0
    vec.extend(present)
    vec.append(other)

    return _as_f32(vec)


def resolve_intent(intent: int, state: dict, bot) -> Action:
    """Resolve a raw intent (0-7) to a concrete :class:`Action`.

    ``bot`` supplies the phase-specific ``_hand`` / ``_shop`` / ``_pack``
    helpers and a generic ``act`` fallback (a HeuristicBot works).
    Pure: reads ``state``, returns an Action; no SimClient involvement.
    """
    phase = state.get("state", "")
    action = None
    if intent == 0 and phase == "BLIND_SELECT":
        action = Action("select", note="take blind")
    elif intent == 1 and phase == "SELECTING_HAND":
        action = bot._hand(state)
    elif intent == 2 and phase == "SELECTING_HAND":
        a = bot._hand(state)
        action = a if a.method == "discard" else None
    elif intent == 3 and phase == "SHOP":
        a = bot._shop(state)
        action = a if a.method == "buy" else None
    elif intent == 4 and phase == "SHOP":
        a = bot._shop(state)
        action = a if a.method == "use" else None
    elif intent == 5 and phase == "SHOP":
        action = Action("next_round", note="leave shop")
    elif intent == 6 and phase == "ROUND_EVAL":
        action = Action("cash_out", note="cash out")
    elif intent == 7 and phase == "SMODS_BOOSTER_OPENED":
        action = bot._pack(state)
    elif intent == 8 and phase == "SHOP":
        action = bot._sell(state)
    elif intent == 9 and phase == "SHOP":
        action = bot._reroll(state)

    if action is None:
        # Intent doesn't line up with the current phase: use the generic
        # phase action from the bot, then the runner fallback if needed.
        # Tag the note so narration can attribute heuristic-driven steps.
        action = bot.act(state)
        action.note = f"[fb] {action.note}" if action.note else "[fb]"
    return action


# -- tiny stdlib stand-ins so the env works even without gymnasium ----------
class _Discrete:
    def __init__(self, n: int):
        self.n = n

    def sample(self):
        import random
        return random.randrange(self.n)

    def __repr__(self):
        return f"Discrete({self.n})"


class _Box:
    def __init__(self, low, high, shape=None, dtype="float32"):
        self.low = low
        self.high = high
        self.shape = shape or (len(low),)
        self.dtype = dtype

    def __repr__(self):
        return f"Box({self.low}, {self.high}, shape={self.shape})"


def _lf(vals):
    try:
        import numpy as np
        return np.asarray(vals, dtype=np.float32)
    except Exception:
        return vals


def _as_f32(vec):
    try:
        import numpy as np
        return np.asarray(vec, dtype=np.float32)
    except Exception:
        return vec


__all__ = ["BalatroEnv", "OBS_DIM", "N_SCALARS", "HAND_SLOTS", "HAND_DIM",
           "JOKER_KEYS", "JOKER_DIM", "_OBS_DOC", "encode_obs", "resolve_intent"]


if __name__ == "__main__":

    env = BalatroEnv(deck="RED", stake="WHITE", seed="selftest")
    obs, info = env.reset(seed=123)
    assert len(obs) == OBS_DIM, f"obs len {len(obs)} != {OBS_DIM}"
    # Validate the outer bounds match the documented layout.
    assert env.observation_space.shape == (OBS_DIM,), env.observation_space.shape

    n_steps = 50
    for i in range(n_steps):
        action = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(action)
        assert len(obs) == OBS_DIM, f"step {i}: obs len {len(obs)} != {OBS_DIM}"
        assert isinstance(r, float), f"step {i}: reward {r!r} not float"
        assert isinstance(terminated, bool), "terminated not bool"
        assert isinstance(truncated, bool), "truncated not bool"
        assert isinstance(info, dict) and "state" in info
        assert "last_score" in info
        if terminated or truncated:
            break

    # A fresh reset restarts cleanly.
    obs, info = env.reset(seed=999)
    assert len(obs) == OBS_DIM
    print("PASS")
    print(f"action_space = Discrete({env.action_space.n}); obs shape = {env.observation_space.shape}")
    print(f"obs dims: {N_SCALARS} scalars + {HAND_DIM} hand + {JOKER_DIM} jokers")
    print(_OBS_DOC.strip())