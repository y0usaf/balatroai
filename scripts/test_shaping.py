"""Check the shaping term is potential-based, and see its scale on real games.

Two things must hold for the shaping to be safe:

1. Over a full episode the shaping contributions telescope to ``-PHI(s0)``,
   which is a constant the policy cannot influence. If they do not, we have
   accidentally written a bonus that changes the objective.
2. Its magnitude has to be small next to the env's own rewards, or it drowns
   the real signal regardless of invariance.

    nix develop -c .venv/bin/python scripts/test_shaping.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def main() -> None:
    from jackdaw.env.game_interface import DirectAdapter
    from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv

    from shaping import PotentialShaper, potential

    gamma = 0.999
    env = BalatroGymnasiumEnv(
        adapter_factory=DirectAdapter, max_steps=3_000,
        seed_prefix="SHAPE", reward_shaping=True,
    )
    rng = np.random.default_rng(0)

    worst_err = 0.0
    base_abs: list[float] = []
    shape_abs: list[float] = []
    episodes = 0

    for _ep in range(20):
        obs, _ = env.reset()
        gs0 = env._inner._adapter.raw_state
        phi0 = potential(gs0)
        shaper = PotentialShaper(gamma, 1.0)
        shaper.reset(gs0)

        discounted_f = 0.0
        discount = 1.0
        for _t in range(3_000):
            a = int(rng.integers(0, max(1, len(env._action_table))))
            _obs, r, term, trunc, _info = env.step(a)
            done = term or trunc
            f = shaper.step(None if done else env._inner._adapter.raw_state, done)
            discounted_f += discount * f
            discount *= gamma
            base_abs.append(abs(r))
            shape_abs.append(abs(f))
            if done:
                break

        # Sum of gamma^t * F_t must telescope to -PHI(s0).
        worst_err = max(worst_err, abs(discounted_f + phi0))
        episodes += 1

    print(f"episodes checked           {episodes}")
    print(f"max |sum(g^t F_t) + PHI0|  {worst_err:.2e}   (0 = exactly potential-based)")
    print(f"mean |env reward|          {np.mean(base_abs):.4f}")
    print(f"mean |shaping term|        {np.mean(shape_abs):.4f}")
    print(f"shaping / env ratio        {np.mean(shape_abs) / max(np.mean(base_abs), 1e-9):.2f}")
    print("OK" if worst_err < 1e-6 else "FAILED: not potential-based")


if __name__ == "__main__":
    main()
