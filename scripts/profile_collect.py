"""Where does rollout collection actually go?

Splits the collect phase into its three costs, measured in isolation:

* ``env.step`` inside a worker (pure jackdaw simulation),
* ``flatten_obs`` + ``encode_action_table`` (per-step numpy encoding),
* the learner's per-group policy forward + sample + ``.cpu()`` sync.

Run::

    nix develop -c .venv/bin/python scripts/profile_collect.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))


def bench_env(n_envs: int = 8, steps: int = 200) -> None:
    from jackdaw.env.game_interface import DirectAdapter
    from jackdaw.env.gymnasium_wrapper import MAX_ACTIONS, BalatroGymnasiumEnv

    from policy_v3 import OBS_DIM, encode_action_table, flatten_obs

    envs = []
    for i in range(n_envs):
        e = BalatroGymnasiumEnv(
            adapter_factory=DirectAdapter,
            max_steps=3_000,
            seed_prefix=f"PROF_{i}",
            reward_shaping=True,
        )
        e.reset()
        envs.append(e)

    obs_row = np.zeros(OBS_DIM, dtype=np.float32)
    acts_row = np.zeros((MAX_ACTIONS, 3), dtype=np.int16)
    rng = np.random.default_rng(0)

    t_step = t_flat = t_enc = 0.0
    n = 0
    for _ in range(steps):
        for e in envs:
            a = int(rng.integers(0, max(1, len(e._action_table))))
            t0 = time.perf_counter()
            obs, _r, term, trunc, _info = e.step(a)
            t1 = time.perf_counter()
            obs_row[:] = flatten_obs(obs)
            t2 = time.perf_counter()
            encode_action_table(e._action_table, acts_row)
            t3 = time.perf_counter()
            t_step += t1 - t0
            t_flat += t2 - t1
            t_enc += t3 - t2
            n += 1
            if term or trunc:
                obs, _ = e.reset()

    total = t_step + t_flat + t_enc
    print(f"env-steps: {n:,} in {total:.2f}s  ->  {n / total:,.0f} steps/s/proc")
    for name, t in (("env.step", t_step), ("flatten_obs", t_flat), ("encode_acts", t_enc)):
        print(f"  {name:<12} {t:6.2f}s  {100 * t / total:5.1f}%  {1e6 * t / n:7.1f} us/step")


def bench_policy(
    batch: int = 120, a_eff: int = 128, iters: int = 300, mode: str | None = None
) -> None:
    from policy_v3 import OBS_DIM, PointerPolicy, masked_dist

    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    policy = PointerPolicy().to(dev)
    fwd = policy
    if mode == "compile":
        fwd = torch.compile(policy, dynamic=False)
    elif mode == "cudagraph":
        fwd = torch.compile(policy, dynamic=False, mode="reduce-overhead")

    x = torch.randn(batch, OBS_DIM, device=dev)
    acts = torch.randint(0, 8, (batch, a_eff, 3), dtype=torch.int16, device=dev)
    nact = torch.full((batch,), a_eff, dtype=torch.long, device=dev)
    act_np = np.zeros(batch, dtype=np.int64)

    with torch.inference_mode():
        for _ in range(30):  # warmup / compilation
            with torch.autocast("cuda", torch.bfloat16):
                logits, v = fwd(x, acts)
            masked_dist(logits.float(), nact).sample()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.autocast("cuda", torch.bfloat16):
                logits, v = fwd(x, acts)
            a = masked_dist(logits.float(), nact).sample()
            act_np[:] = a.cpu().numpy()  # the sync the trainer pays every step
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    print(f"[{mode or 'eager':<10}] batch {batch:>4}, a_eff {a_eff:>3} -> "
          f"{1e3 * dt / iters:5.2f} ms/call   "
          f"per iter (128x3 groups): {384 * dt / iters:.2f}s", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "policy"):
        bench_policy(mode=None)
        bench_policy(mode="compile")
        bench_policy(mode="cudagraph")
    if which in ("all", "env"):
        bench_env()
