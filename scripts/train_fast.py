"""Async pipelined PPO for the jackdaw Balatro sim — built for throughput.

Architecture:

- N worker processes, each owning K envs, communicating through shared
  memory (obs = flat float32 vector, mask = one int: the action table is
  a dense prefix).  No pickling, no pipes on the hot path.
- Workers are split into groups; while group A's envs step, the learner
  runs GPU inference for group B (double buffering) — CPU and GPU stay
  busy simultaneously.
- CleanRL-style PPO update on GPU with the entity-attention policy.

Usage::

    LD_LIBRARY_PATH=/run/opengl-driver/lib \
        uv run python scripts/train_fast.py --total-steps 100000000

Monitor:  uv run tensorboard --logdir runs/
"""

from __future__ import annotations

import argparse
import atexit
import multiprocessing as mp
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def worker_proc(
    rank: int,
    k: int,
    n_total: int,
    shm_names: dict[str, str],
    act_ev,
    obs_ev,
    stats_q,
    max_steps: int,
) -> None:
    from jackdaw.env.game_interface import DirectAdapter
    from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv

    from policy import OBS_DIM, flatten_obs

    shms = {n: shared_memory.SharedMemory(name=s) for n, s in shm_names.items()}
    obs_buf = np.ndarray((n_total, OBS_DIM), dtype=np.float32, buffer=shms["obs"].buf)
    nact_buf = np.ndarray((n_total,), dtype=np.int32, buffer=shms["nact"].buf)
    act_buf = np.ndarray((n_total,), dtype=np.int64, buffer=shms["act"].buf)
    rew_buf = np.ndarray((n_total,), dtype=np.float32, buffer=shms["rew"].buf)
    done_buf = np.ndarray((n_total,), dtype=np.float32, buffer=shms["done"].buf)

    lo = rank * k
    envs = []
    for i in range(k):
        env = BalatroGymnasiumEnv(
            adapter_factory=DirectAdapter,
            max_steps=max_steps,
            seed_prefix=f"F{rank}_{i}",
            reward_shaping=True,
        )
        obs, _ = env.reset()
        obs_buf[lo + i] = flatten_obs(obs)
        nact_buf[lo + i] = len(env._action_table)
        envs.append(env)
    obs_ev.set()

    while True:
        act_ev.wait()
        act_ev.clear()
        for i, env in enumerate(envs):
            gi = lo + i
            obs, r, term, trunc, info = env.step(int(act_buf[gi]))
            done = term or trunc
            if done:
                stats_q.put(
                    (
                        info.get("balatro/ante_reached", 1),
                        info.get("balatro/rounds_beaten", 0),
                        bool(info.get("balatro/won", False)),
                    )
                )
                obs, _ = env.reset()
            obs_buf[gi] = flatten_obs(obs)
            nact_buf[gi] = len(env._action_table)
            rew_buf[gi] = r
            done_buf[gi] = 1.0 if done else 0.0
        obs_ev.set()


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Async pipelined PPO on Balatro")
    parser.add_argument("--total-steps", type=int, default=100_000_000)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--envs-per-worker", type=int, default=8)
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--minibatch", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=3_000)
    parser.add_argument("--log-dir", type=str, default="runs/fast")
    parser.add_argument("--checkpoint-every", type=int, default=2_000_000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from torch.utils.tensorboard import SummaryWriter

    from policy import OBS_DIM, PolicyNet, masked_dist

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    n_env = args.workers * args.envs_per_worker
    T = args.rollout
    log_path = Path(args.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_path))

    # -- shared memory ------------------------------------------------------
    specs = {
        "obs": n_env * OBS_DIM * 4,
        "nact": n_env * 4,
        "act": n_env * 8,
        "rew": n_env * 4,
        "done": n_env * 4,
    }
    shms = {n: shared_memory.SharedMemory(create=True, size=s) for n, s in specs.items()}

    def _cleanup() -> None:
        for s in shms.values():
            try:
                s.close()
                s.unlink()
            except Exception:
                pass

    atexit.register(_cleanup)

    obs_buf = np.ndarray((n_env, OBS_DIM), dtype=np.float32, buffer=shms["obs"].buf)
    nact_buf = np.ndarray((n_env,), dtype=np.int32, buffer=shms["nact"].buf)
    act_buf = np.ndarray((n_env,), dtype=np.int64, buffer=shms["act"].buf)
    rew_buf = np.ndarray((n_env,), dtype=np.float32, buffer=shms["rew"].buf)
    done_buf = np.ndarray((n_env,), dtype=np.float32, buffer=shms["done"].buf)

    # -- workers -------------------------------------------------------------
    ctx = mp.get_context("spawn")
    stats_q = ctx.Queue()
    act_evs = [ctx.Event() for _ in range(args.workers)]
    obs_evs = [ctx.Event() for _ in range(args.workers)]
    shm_names = {n: s.name for n, s in shms.items()}
    procs = []
    for w in range(args.workers):
        p = ctx.Process(
            target=worker_proc,
            args=(w, args.envs_per_worker, n_env, shm_names,
                  act_evs[w], obs_evs[w], stats_q, args.max_steps),
            daemon=True,
        )
        p.start()
        procs.append(p)

    # worker groups → contiguous env-index blocks
    groups: list[list[int]] = [
        list(range(g, args.workers, args.groups)) for g in range(args.groups)
    ]
    group_idx_np = [
        np.concatenate([np.arange(w * args.envs_per_worker,
                                  (w + 1) * args.envs_per_worker) for w in g])
        for g in groups
    ]
    group_idx = [torch.as_tensor(i, device=device) for i in group_idx_np]

    # -- model ----------------------------------------------------------------
    policy = PolicyNet().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        policy.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        global_step = ckpt["step"]
        print(f"resumed from {args.resume} at step {global_step:,}")

    # -- rollout storage (GPU) -------------------------------------------------
    obs_s = torch.zeros((T, n_env, OBS_DIM), device=device)
    nact_s = torch.zeros((T, n_env), dtype=torch.long, device=device)
    act_s = torch.zeros((T, n_env), dtype=torch.long, device=device)
    logp_s = torch.zeros((T, n_env), device=device)
    val_s = torch.zeros((T, n_env), device=device)
    rew_s = torch.zeros((T, n_env), device=device)
    done_s = torch.zeros((T, n_env), device=device)

    # prime: wait for all workers' reset obs
    for ev in obs_evs:
        ev.wait()
        ev.clear()
    last_obs = torch.as_tensor(obs_buf.copy(), device=device)
    last_n = torch.as_tensor(nact_buf.astype(np.int64), device=device)

    ep_antes: list[int] = []
    ep_rounds: list[int] = []
    ep_wins: list[bool] = []
    next_ckpt = (global_step // args.checkpoint_every + 1) * args.checkpoint_every
    print(f"{args.workers} workers x {args.envs_per_worker} envs = {n_env} envs, "
          f"{args.groups} pipeline groups, rollout {T} "
          f"({T * n_env:,} steps/iter)")

    while global_step < args.total_steps:
        t_start = time.time()

        # ---- collect (pipelined across groups) ----
        with torch.inference_mode():
            for t in range(T):
                # inference + dispatch per group; while group g steps on
                # CPU, the next group's forward runs on GPU
                for gi, g in enumerate(groups):
                    idx, idx_np = group_idx[gi], group_idx_np[gi]
                    o = last_obs[idx]
                    with torch.autocast("cuda", torch.bfloat16):
                        logits, v = policy(o)
                    logits, v = logits.float(), v.float()
                    dist = masked_dist(logits, last_n[idx])
                    a = dist.sample()
                    obs_s[t, idx] = o
                    nact_s[t, idx] = last_n[idx]
                    act_s[t, idx] = a
                    logp_s[t, idx] = dist.log_prob(a)
                    val_s[t, idx] = v
                    act_buf[idx_np] = a.cpu().numpy()
                    for w in g:
                        act_evs[w].set()
                # harvest step results per group
                for gi, g in enumerate(groups):
                    for w in g:
                        obs_evs[w].wait()
                        obs_evs[w].clear()
                    idx, idx_np = group_idx[gi], group_idx_np[gi]
                    rew_s[t, idx] = torch.as_tensor(rew_buf[idx_np].copy(), device=device)
                    done_s[t, idx] = torch.as_tensor(done_buf[idx_np].copy(), device=device)
                    last_obs[idx] = torch.as_tensor(obs_buf[idx_np].copy(), device=device)
                    last_n[idx] = torch.as_tensor(
                        nact_buf[idx_np].astype(np.int64), device=device)

            # ---- GAE ----
            with torch.autocast("cuda", torch.bfloat16):
                _, boot_v = policy(last_obs)
            boot_v = boot_v.float()
            adv = torch.zeros_like(rew_s)
            last_gae = torch.zeros(n_env, device=device)
            for t in reversed(range(T)):
                nonterm = 1.0 - done_s[t]
                next_v = boot_v if t == T - 1 else val_s[t + 1]
                delta = rew_s[t] + args.gamma * next_v * nonterm - val_s[t]
                last_gae = delta + args.gamma * args.gae_lambda * nonterm * last_gae
                adv[t] = last_gae
            ret = adv + val_s

        global_step += T * n_env
        t_collect = time.time() - t_start

        # ---- PPO update ----
        b_obs = obs_s.reshape(-1, OBS_DIM)
        b_nact = nact_s.reshape(-1)
        b_act = act_s.reshape(-1)
        b_logp = logp_s.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = ret.reshape(-1)
        batch = T * n_env
        pg_l = vf_l = ent_l = kl = 0.0
        n_upd = 0
        for _ in range(args.epochs):
            perm = torch.randperm(batch, device=device)
            for s in range(0, batch, args.minibatch):
                mb = perm[s : s + args.minibatch]
                with torch.autocast("cuda", torch.bfloat16):
                    logits, v = policy(b_obs[mb])
                logits, v = logits.float(), v.float()
                dist = masked_dist(logits, b_nact[mb])
                logp = dist.log_prob(b_act[mb])
                ratio = (logp - b_logp[mb]).exp()
                madv = b_adv[mb]
                madv = (madv - madv.mean()) / (madv.std() + 1e-8)
                pg = torch.max(
                    -madv * ratio,
                    -madv * ratio.clamp(1 - args.clip, 1 + args.clip),
                ).mean()
                vf = 0.5 * (v - b_ret[mb]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pg + args.vf_coef * vf - args.ent_coef * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    pg_l += pg.item(); vf_l += vf.item(); ent_l += ent.item()
                    kl += (b_logp[mb] - logp).mean().item()
                n_upd += 1

        # ---- metrics ----
        while not stats_q.empty():
            a, r, won = stats_q.get_nowait()
            ep_antes.append(a); ep_rounds.append(r); ep_wins.append(won)
        t_iter = time.time() - t_start
        sps = int(T * n_env / t_iter)
        writer.add_scalar("perf/sps", sps, global_step)
        writer.add_scalar("perf/collect_s", t_collect, global_step)
        writer.add_scalar("perf/update_s", t_iter - t_collect, global_step)
        writer.add_scalar("loss/policy", pg_l / n_upd, global_step)
        writer.add_scalar("loss/value", vf_l / n_upd, global_step)
        writer.add_scalar("loss/entropy", ent_l / n_upd, global_step)
        writer.add_scalar("loss/approx_kl", kl / n_upd, global_step)
        line = (f"step {global_step:>12,}  sps {sps:>6,}  "
                f"c/u {t_collect:.1f}/{t_iter - t_collect:.1f}s")
        if ep_antes:
            writer.add_scalar("balatro/mean_ante", np.mean(ep_antes), global_step)
            writer.add_scalar("balatro/max_ante", np.max(ep_antes), global_step)
            writer.add_scalar("balatro/mean_rounds", np.mean(ep_rounds), global_step)
            writer.add_scalar("balatro/win_rate", np.mean(ep_wins), global_step)
            line += (f"  ante {np.mean(ep_antes):.2f} (max {np.max(ep_antes)})"
                     f"  win {100 * np.mean(ep_wins):.1f}%"
                     f"  eps {len(ep_antes)}")
            ep_antes.clear(); ep_rounds.clear(); ep_wins.clear()
        print(line, flush=True)

        # ---- checkpoint ----
        if global_step >= next_ckpt:
            ck = {"model": policy.state_dict(), "opt": opt.state_dict(),
                  "step": global_step}
            torch.save(ck, log_path / f"ckpt_{global_step}.pt")
            torch.save(ck, log_path / "latest.pt")
            next_ckpt += args.checkpoint_every

    torch.save({"model": policy.state_dict(), "opt": opt.state_dict(),
                "step": global_step}, log_path / "final.pt")
    print(f"done — saved {log_path / 'final.pt'}")


if __name__ == "__main__":
    main()
