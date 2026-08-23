"""Evaluate a trained policy AND its reward function's alignment.

Runs episodes through BalatroEnv (same code path as training) and
reports:
  - performance: avg ante, win rate
  - alignment:   Spearman correlation between episode return and ante
                 reached.  A well-shaped reward makes return predict
                 success; near-zero means the objective is misaligned
                 with what we actually want.
  - balance:     per-term contribution averages (from env info), so
                 dominant or dead terms are visible.

Usage:
  python scripts/reward_eval.py [CHECKPOINT] [GAMES]
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from balatroai.bots.heuristic import HeuristicBot
from balatroai.rl.train import load_model

ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/masked_v2/mpo_latest"
games = int(sys.argv[2]) if len(sys.argv) > 2 else 30

model = load_model(ckpt, device="cpu")
masked = type(model).__name__ == "MaskablePPO"
with open(ckpt + "_vecnormalize.pkl", "rb") as f:
    rms = pickle.load(f).obs_rms

bot = HeuristicBot()
env = __import__("balatroai.rl.env", fromlist=["BalatroEnv"]).BalatroEnv()

v_returns, v_antes, v_wins = [], [], []
v_terms: dict[str, float] = {}
v_steps = 0

for i in range(games):
    env.reset(seed=i)
    ep_ret, ante = 0.0, 1
    won = False
    for _ in range(4000):
        obs = np.asarray(env.obs(), dtype=np.float32)
        obs = np.clip((obs - rms.mean) / np.sqrt(rms.var + 1e-8), -10, 10)
        kw = {"action_masks": [env.action_masks()]} if masked else {}
        intent = int(model.predict(obs, deterministic=True, **kw)[0])
        obs, r, term, trunc, info = env.step(intent)
        ep_ret += float(r)
        v_steps += 1
        for k, v in (info.get("reward_terms") or {}).items():
            v_terms[k] = v_terms.get(k, 0.0) + float(v)
        ante = max(ante, info["state"].get("ante_num", 0))
        if term or trunc:
            won = bool(info["state"].get("won"))
            break
    v_returns.append(ep_ret)
    v_antes.append(ante)
    if won:
        v_wins.append(1)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


rho = spearman(np.array(v_returns), np.array(v_antes))
print(f"checkpoint={ckpt} games={games}")
print(f"performance : avg_ante={np.mean(v_antes):.2f} "
      f"max={max(v_antes)} won={len(v_wins)}/{games}")
print(f"episode ret : mean={np.mean(v_returns):+.1f} std={np.std(v_returns):.1f}")
print(f"alignment   : spearman(return, ante_reached) = {rho:+.2f} "
      f"(near 0 = reward not tracking success)")
print("balance     : per-step term averages")
for k in sorted(v_terms):
    print(f"  {k:>15}: {v_terms[k] / max(v_steps, 1):+.4f}")
