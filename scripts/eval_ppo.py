"""Evaluate a trained MaskablePPO model on the jackdaw sim.

Usage::

    uv run python scripts/eval_ppo.py runs/ppo/best_model.zip --episodes 200
"""

from __future__ import annotations

import argparse

import numpy as np
from sb3_contrib import MaskablePPO

from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Balatro PPO model")
    parser.add_argument("model", help="path to model .zip")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed-prefix", type=str, default="HOLDOUT")
    parser.add_argument("--stochastic", action="store_true",
                        help="sample the policy instead of argmax")
    args = parser.parse_args()

    model = MaskablePPO.load(args.model, device="cpu")
    env = BalatroGymnasiumEnv(
        adapter_factory=DirectAdapter,
        max_steps=3_000,
        seed_prefix=args.seed_prefix,
        reward_shaping=False,
    )

    antes: list[int] = []
    rounds: list[int] = []
    wins: list[bool] = []
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(
                obs, action_masks=mask, deterministic=not args.stochastic
            )
            obs, _, term, trunc, info = env.step(int(action))
            done = term or trunc
        antes.append(info["balatro/ante_reached"])
        rounds.append(info["balatro/rounds_beaten"])
        wins.append(info["balatro/won"])
        print(
            f"ep {ep + 1:4d}/{args.episodes}  "
            f"{'W' if wins[-1] else 'L'}  ante {antes[-1]}  rounds {rounds[-1]}"
        )

    a = np.array(antes)
    print(
        f"\n{args.model}: {sum(wins)}/{args.episodes} wins "
        f"({100 * np.mean(wins):.1f}%) · ante avg {a.mean():.2f} · "
        f"median {int(np.median(a))} · best {a.max()}"
    )


if __name__ == "__main__":
    main()
