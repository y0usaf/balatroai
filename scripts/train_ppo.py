"""Train MaskablePPO on the jackdaw Balatro sim — tuned for real hardware.

Improvements over jackdaw's reference script:

- SubprocVecEnv parallel rollout across CPU cores
- larger policy/value networks (the reference uses SB3 defaults)
- long-horizon discounting (gamma=0.999 — episodes run 100s of steps)
- periodic masked eval on held-out seeds, best-model checkpointing
- resumable (--resume path/to/model.zip)

Usage (from the balatroAI repo, needs `uv sync --extra train`)::

    LD_LIBRARY_PATH=/run/opengl-driver/lib \
        uv run python scripts/train_ppo.py --total-timesteps 20000000

Monitor::

    uv run tensorboard --logdir runs/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv


class BalatroMetricsCallback(BaseCallback):
    """Log Balatro-specific episode metrics to tensorboard."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._antes: list[int] = []
        self._rounds: list[int] = []
        self._wins: list[bool] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "balatro/ante_reached" in info:
                self._antes.append(info["balatro/ante_reached"])
                self._rounds.append(info["balatro/rounds_beaten"])
                self._wins.append(info["balatro/won"])
        return True

    def _on_rollout_end(self) -> None:
        if not self._antes:
            return
        self.logger.record("balatro/mean_ante_reached", np.mean(self._antes))
        self.logger.record("balatro/max_ante_reached", np.max(self._antes))
        self.logger.record("balatro/mean_rounds_beaten", np.mean(self._rounds))
        self.logger.record("balatro/win_rate", np.mean(self._wins))
        self._antes.clear()
        self._rounds.clear()
        self._wins.clear()


def make_env(rank: int, seed_prefix: str, max_steps: int):
    def _init() -> BalatroGymnasiumEnv:
        return BalatroGymnasiumEnv(
            adapter_factory=DirectAdapter,
            max_steps=max_steps,
            seed_prefix=f"{seed_prefix}_{rank}",
            reward_shaping=True,
        )

    return _init


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MaskablePPO on Balatro (jackdaw sim)")
    parser.add_argument("--total-timesteps", type=int, default=20_000_000)
    parser.add_argument("--n-envs", type=int, default=24)
    parser.add_argument("--log-dir", type=str, default="runs/ppo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=3_000)
    parser.add_argument("--resume", type=str, default=None,
                        help="path to a saved model .zip to continue training")
    parser.add_argument("--arch", choices=["mlp", "attention"], default="mlp",
                        help="mlp = flat MultiInputPolicy; attention = entity transformer")
    args = parser.parse_args()

    log_path = Path(args.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    env = SubprocVecEnv(
        [make_env(i, "TRAIN", args.max_steps) for i in range(args.n_envs)]
    )
    eval_env = SubprocVecEnv([make_env(1000, "EVAL", args.max_steps)])

    if args.arch == "attention":
        from extractor import EntityAttentionExtractor

        policy_kwargs = dict(
            features_extractor_class=EntityAttentionExtractor,
            features_extractor_kwargs=dict(
                d_model=128, n_heads=4, n_layers=2, features_dim=256
            ),
            net_arch=dict(pi=[256], vf=[256]),
        )
    else:
        policy_kwargs = dict(net_arch=dict(pi=[512, 512], vf=[512, 512]))

    if args.resume:
        model = MaskablePPO.load(args.resume, env=env, device="cuda")
        print(f"resumed from {args.resume} at {model.num_timesteps} timesteps")
    else:
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            seed=args.seed,
            device="cuda",
            tensorboard_log=str(log_path),
            policy_kwargs=policy_kwargs,
            learning_rate=2.5e-4,
            n_steps=1024,          # per env → 24k-step rollouts at n_envs=24
            batch_size=4096,
            n_epochs=4,
            gamma=0.999,           # long episodes: a win is ~500+ steps out
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )

    callbacks = [
        BalatroMetricsCallback(),
        CheckpointCallback(
            save_freq=max(1_000_000 // args.n_envs, 1),
            save_path=str(log_path / "checkpoints"),
            name_prefix="ppo",
        ),
        MaskableEvalCallback(
            eval_env,
            n_eval_episodes=20,
            eval_freq=max(500_000 // args.n_envs, 1),
            best_model_save_path=str(log_path),
            log_path=str(log_path),
            deterministic=True,
        ),
    ]

    print(f"training for {args.total_timesteps:,} timesteps on {args.n_envs} envs...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        reset_num_timesteps=not args.resume,
    )

    model.save(str(log_path / "final"))
    print(f"saved to {log_path / 'final.zip'}")


if __name__ == "__main__":
    main()
