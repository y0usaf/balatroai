"""Evaluate a train_fast_v3.py (pointer policy) checkpoint.

Usage::

    uv run python -u scripts/eval_fast_v3.py runs/pointer_v3/latest.pt --episodes 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.gymnasium_wrapper import MAX_ACTIONS, BalatroGymnasiumEnv

from policy_v3 import PointerPolicy, encode_action_table, flatten_obs, masked_dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed-prefix", type=str, default="HOLDOUT")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    # Pre-net_cfg checkpoints are all the d=128/2-layer default.
    policy = PointerPolicy(**ckpt.get(
        "net_cfg", {"d_model": 128, "n_heads": 4, "n_layers": 2}))
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    print(f"{args.model} @ step {ckpt.get('step', '?'):,}")

    env = BalatroGymnasiumEnv(
        adapter_factory=DirectAdapter,
        max_steps=3_000,
        seed_prefix=args.seed_prefix,
        reward_shaping=False,
    )
    struct = np.zeros((MAX_ACTIONS, 3), dtype=np.int16)

    antes: list[int] = []
    rounds: list[int] = []
    wins: list[bool] = []
    with torch.inference_mode():
        for ep in range(args.episodes):
            obs, info = env.reset()
            done = False
            while not done:
                x = torch.as_tensor(flatten_obs(obs)).unsqueeze(0)
                n = len(env._action_table)
                encode_action_table(env._action_table, struct)
                acts = torch.as_tensor(struct).unsqueeze(0)
                logits, _ = policy(x, acts)
                if args.stochastic:
                    a = masked_dist(logits, torch.as_tensor([n])).sample().item()
                else:
                    a = logits[0, :n].argmax().item()
                obs, _, term, trunc, info = env.step(int(a))
                done = term or trunc
            antes.append(info["balatro/ante_reached"])
            rounds.append(info["balatro/rounds_beaten"])
            wins.append(info["balatro/won"])
            print(f"ep {ep + 1:4d}/{args.episodes}  "
                  f"{'W' if wins[-1] else 'L'}  ante {antes[-1]}  rounds {rounds[-1]}")

    a = np.array(antes)
    print(f"\n{sum(wins)}/{args.episodes} wins ({100 * np.mean(wins):.1f}%) · "
          f"ante avg {a.mean():.2f} · median {int(np.median(a))} · best {a.max()}")


if __name__ == "__main__":
    main()
