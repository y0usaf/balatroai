"""Bot that drives the live game from a trained RL checkpoint."""
from __future__ import annotations

from ..bots import Action, register
from ..bots.heuristic import HeuristicBot
from .env import encode_obs, resolve_intent


@register
class PolicyBot:
    name = "policy"

    def __init__(self, checkpoint: str = "checkpoints/ppo_run1_latest"):
        import pickle
        from stable_baselines3 import PPO
        self._bot = HeuristicBot()
        # CPU inference: MlpPolicy forward passes are faster there, and watch
        # runs keep the GPU free.
        self._model = PPO.load(checkpoint, device="cpu")
        with open(checkpoint + "_vecnormalize.pkl", "rb") as f:
            data = pickle.load(f)
        # Train.py persists a VecNormalize object directly via env.save().
        # Its observation stats live on ``obs_rms`` (a RunningMeanStd with
        # .mean/.var); clip_obs matches the VecNormalize default the training
        # used (10.0).  We mirror VecNormalize.normalize_obs exactly.
        self._obs_rms = data.obs_rms
        self._clip_obs = float(getattr(data, "clip_obs", 10.0))

    def act(self, state: dict) -> Action:
        import numpy as np
        obs = np.asarray(encode_obs(state), dtype=np.float32)
        mean = self._obs_rms.mean
        var = self._obs_rms.var
        obs = (obs - mean) / np.sqrt(var + 1e-8)
        obs = np.clip(obs, -self._clip_obs, self._clip_obs)
        intent, _ = self._model.predict(obs, deterministic=True)
        return resolve_intent(int(intent), state, self._bot)
