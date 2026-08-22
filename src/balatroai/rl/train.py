"""PPO training over BalatroEnv (P3). torch/sb3 imported lazily so the
stdlib-only core stays intact."""
from __future__ import annotations

from collections import Counter


SUITS = {"H": "♥", "D": "♦", "C": "♣", "S": "♠"}


def _fmt_card(c: dict) -> str:
    v = c.get("value", {})
    rank, suit = v.get("rank"), v.get("suit")
    if rank and suit:
        return f"{rank}{SUITS.get(suit, suit)}"
    return c.get("label", "?")


def load_model(checkpoint: str, env=None, device: str | None = None):
    """Load a PPO or MaskablePPO checkpoint as its right class.

    SB3 zips don't record their algorithm class, so try the masked loader
    first and fall back to plain PPO for pre-masking checkpoints.
    """
    from stable_baselines3 import PPO
    try:
        from sb3_contrib import MaskablePPO
    except ImportError:  # train extra without sb3-contrib: legacy only
        MaskablePPO = None
    kwargs = {} if device is None else {"device": device}
    if MaskablePPO is not None:
        try:
            return MaskablePPO.load(checkpoint, env=env, **kwargs)
        except (TypeError, ValueError):
            pass  # unmasked checkpoint: its policy rejects the masked ctor
    return PPO.load(checkpoint, env=env, **kwargs)


def train(workers: int = 4, timesteps: int = 100_000, checkpoint: str = "checkpoint",
          eval_freq: int = 5000, n_eval_games: int = 20) -> None:
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecNormalize,
        sync_envs_normalization,
    )

    from .env import BalatroEnv

    def make_env():
        return BalatroEnv()

    env = SubprocVecEnv([make_env for _ in range(workers)])
    # Normalize obs + rewards to ~unit scale: raw obs (money, chips, blind
    # score) and rewards (chips_delta/1000 can hit 1000+) have huge ranges
    # that the value head cannot fit.  Stats must be saved alongside the
    # model, or a reloaded policy sees wrong-scaled inputs.
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Eval env: single game, normalized with the training env's stats, but
    # rewards left raw so "mean reward" is interpretable.
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

    class EvalAndLogCallback(BaseCallback):
        """Periodically eval the policy: log mean reward, avg ante, and the
        action (intent) histogram so game progress is observable, not just
        value loss."""

        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.eval_freq = eval_freq
            self.n_games = n_eval_games

        def _on_step(self) -> bool:
            if self.n_calls % self.eval_freq != 0:
                return True
            sync_envs_normalization(env, eval_env)
            rewards: list[float] = []
            antes: list[float] = []
            actions: Counter[str] = Counter()
            terms: dict[str, float] = {}
            term_steps = 0
            for _ in range(self.n_games):
                obs = eval_env.reset()
                done = False
                ep_reward = 0.0
                while not done:
                    masks = eval_env.env_method("action_masks")
                    action, _ = self.model.predict(
                        obs, deterministic=True, action_masks=masks)
                    obs, r, d, info = eval_env.step(action)
                    ep_reward += float(r[0])
                    actions[info[0].get("action_method", "?")] += 1
                    for name, v in (info[0].get("reward_terms") or {}).items():
                        terms[name] = terms.get(name, 0.0) + float(v)
                    term_steps += 1
                    done = bool(d[0])
                antes.append(float(info[0].get("state", {}).get("ante_num", 0)))
                rewards.append(ep_reward)
            self.logger.record("eval/mean_reward", sum(rewards) / len(rewards))
            self.logger.record("eval/avg_ante", sum(antes) / len(antes))
            for name, c in sorted(actions.items()):
                self.logger.record(f"eval/action_{name}", c)
            for name, total in sorted(terms.items()):
                self.logger.record(f"eval/term_{name}", total / max(term_steps, 1))
            # rolling checkpoint so the policy can be watched mid-training
            self.model.save(checkpoint + "_latest")
            env.save(checkpoint + "_latest_vecnormalize.pkl")
            return True

    # MlpPolicy PPO trains faster on CPU than on GPU (sb3#1245).
    model = MaskablePPO("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=timesteps, callback=EvalAndLogCallback())
    model.save(checkpoint)
    env.save(checkpoint + "_vecnormalize.pkl")
    env.close()
    print(f"saved checkpoint: {checkpoint}")


def watch_checkpoint(checkpoint: str, games: int = 1, seed: str | None = None) -> None:
    """Load a trained policy and play it on the sim with per-step narration."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from .env import BalatroEnv

    env = DummyVecEnv([lambda: BalatroEnv(seed=seed)])
    env = VecNormalize.load(checkpoint + "_vecnormalize.pkl", env)
    model = load_model(checkpoint, env=env)
    masked = type(model).__name__ == "MaskablePPO"

    for g in range(games):
        obs = env.reset()
        done = False
        print(f"--- game {g + 1} ---")
        while not done:
            kwargs = ({"action_masks": env.env_method("action_masks")}
                      if masked else {})
            action, _ = model.predict(obs, deterministic=True, **kwargs)
            obs, r, d, info = env.step(action)
            st = info[0]["state"]
            method = info[0]["action_method"]
            ante = st.get("ante_num", "?")
            rnd_num = st.get("round_num", "?")
            money = st.get("money", 0)
            rnd = st.get("round", {})
            chips = rnd.get("chips", 0)
            hands = rnd.get("hands_left", "?")
            disc = rnd.get("discards_left", "?")
            line = f"[a{ante} r{rnd_num} ${money}] {method} → {chips} chips · {hands}h/{disc}d"
            if st.get("state") == "SELECTING_HAND":
                cards = st.get("hand", {}).get("cards", [])
                line += "\n    hand: " + " ".join(_fmt_card(c) for c in cards)
            print(line)
            done = bool(d[0])
        print(f"GAME OVER — ante {st.get('ante_num')}, won={st.get('won')}")
