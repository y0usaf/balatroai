"""Restart training episodes from recorded mid-game states.

Why: a from-scratch policy dies in ante 1, so shops and jokers are effectively
unreachable and the synergies we want it to learn never appear in its data.
Injecting recorded positions makes those situations common.

The obvious risk is that the agent memorizes a fixed set of boards. Three
things prevent that, and the trainer reports the numbers to prove it:

* the pool spans hundreds of distinct seeds, sharded across workers;
* :func:`inject` reseeds the game's PRNG, so replaying one snapshot produces
  different draws, different shop stock and different boss blinds every time.
  The position is reused; the game never is;
* only a fraction of episodes are injected, and episode statistics are tracked
  separately for fresh versus injected starts -- so if the policy is only good
  on injected boards, the fresh-start number stays flat and says so.
"""

from __future__ import annotations

import copy
import pickle
import random
import string
from pathlib import Path
from typing import Any


class UnusableSnapshot(RuntimeError):
    """Raised when a recorded state cannot be resumed as a playable episode."""


def load_pool(path: str | Path, shard: int = 0, n_shards: int = 1) -> list[dict]:
    """Load a curriculum pool, optionally taking one worker's shard.

    Each worker holds a slice rather than the whole pool, so N workers cost one
    copy of the pool in total instead of N copies.
    """
    with Path(path).open("rb") as f:
        pool: list[dict] = pickle.load(f)
    if n_shards > 1:
        pool = pool[shard::n_shards]
    return pool


def _fresh_seed(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_uppercase + string.digits, k=8))


def inject(env: Any, snapshot: dict, rng: random.Random) -> tuple[Any, dict]:
    """Restart ``env`` from ``snapshot`` with fresh randomness.

    Mirrors what ``BalatroEnvironment.reset`` does after the adapter is
    initialized, so the wrapper's bookkeeping (action table, episode counters,
    reward trackers) matches a normal reset.
    """
    from jackdaw.engine.rng import PseudoRandom
    from jackdaw.env.action_space import get_action_mask
    from jackdaw.env.balatro_env import _action_mask_to_game, _compute_shop_splits
    from jackdaw.env.observation import encode_observation

    inner = env._inner
    gs = copy.deepcopy(snapshot["gs"])

    # Same board, different future: a new PRNG means new draws, new shop stock
    # and new boss blinds, so one snapshot cannot be memorized as one game.
    gs["rng"] = PseudoRandom(_fresh_seed(rng))

    inner._adapter._gs = gs
    inner._step_count = 0
    inner.episode_length = 0
    inner.episode_won = False
    inner.episode_ante = int(snapshot.get("ante", 1))

    obs = encode_observation(gs)
    mask = get_action_mask(gs)
    info: dict[str, Any] = {
        "raw_state": gs,
        "shop_splits": _compute_shop_splits(gs),
        "observation": obs,
        "action_mask": mask,
    }
    game_obs = obs.to_game_observation()
    game_mask = _action_mask_to_game(mask)

    # Wrapper-level state, mirroring BalatroGymnasiumEnv.reset.
    env._prev_ante = inner.episode_ante
    env._prev_round = gs.get("round", 0)
    env._prev_chips = gs.get("chips", 0)
    env._episode_max_ante = inner.episode_ante
    env._episode_max_round = gs.get("round", 0)
    env._action_table = env._enumerate_actions(game_mask, info)
    if not env._action_table:
        # A few recorded positions have no legal action once resumed (they were
        # captured on a phase boundary). Callers fall back to a fresh start;
        # stepping such a state would index an empty action table and crash.
        raise UnusableSnapshot(f"no legal actions at ante {snapshot.get('ante')}")
    return env._build_obs(game_obs), info
