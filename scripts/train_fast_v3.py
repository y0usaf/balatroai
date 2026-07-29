"""Async pipelined PPO with the pointer policy (v3).

Same shared-memory/pipelined-groups architecture as train_fast.py, plus an
``acts`` buffer: workers encode each step's action table as (A, 3) int16
rows [action_type, entity_target, card_bitmask] so the learner can score
actions by content (see policy_v3.py).

Usage::

    LD_LIBRARY_PATH=/run/opengl-driver/lib \
        uv run python scripts/train_fast_v3.py --total-steps 200000000
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
    gamma: float,
    shaping_coef: float,
    n_workers: int,
    curriculum_pool: str | None,
    curriculum_frac: float,
) -> None:
    import random as _random

    from jackdaw.env.game_interface import DirectAdapter
    from jackdaw.env.gymnasium_wrapper import MAX_ACTIONS, BalatroGymnasiumEnv

    from policy_v3 import OBS_DIM, encode_action_table, flatten_obs
    from shaping import PotentialShaper

    # Each worker keeps only its shard, so the pool costs one copy in total
    # rather than one per worker.
    pool = []
    if curriculum_pool and curriculum_frac > 0:
        from curriculum import UnusableSnapshot, inject, load_pool
        pool = load_pool(curriculum_pool, shard=rank, n_shards=n_workers)
    cur_rng = _random.Random(1234 + rank)

    shms = {n: shared_memory.SharedMemory(name=s) for n, s in shm_names.items()}
    obs_buf = np.ndarray((n_total, OBS_DIM), dtype=np.float32, buffer=shms["obs"].buf)
    nact_buf = np.ndarray((n_total,), dtype=np.int32, buffer=shms["nact"].buf)
    act_buf = np.ndarray((n_total,), dtype=np.int64, buffer=shms["act"].buf)
    rew_buf = np.ndarray((n_total,), dtype=np.float32, buffer=shms["rew"].buf)
    done_buf = np.ndarray((n_total,), dtype=np.float32, buffer=shms["done"].buf)
    acts_buf = np.ndarray(
        (n_total, MAX_ACTIONS, 3), dtype=np.int16, buffer=shms["acts"].buf)

    lo = rank * k
    envs = []
    shapers = []
    was_injected = [False] * k

    def start_episode(env, i: int):
        """Begin an episode, sometimes from a recorded mid-game position."""
        obs = None
        if pool and cur_rng.random() < curriculum_frac:
            try:
                obs, _ = inject(env, cur_rng.choice(pool), cur_rng)
                was_injected[i] = True
            except UnusableSnapshot:
                obs = None
        if obs is None:
            obs, _ = env.reset()
            was_injected[i] = False
        if shaping_coef:
            shapers[i].reset(env._inner._adapter.raw_state)
        return obs

    for i in range(k):
        env = BalatroGymnasiumEnv(
            adapter_factory=DirectAdapter,
            max_steps=max_steps,
            seed_prefix=f"P{rank}_{i}",
            reward_shaping=True,
        )
        envs.append(env)
        shapers.append(PotentialShaper(gamma, shaping_coef))
        obs = start_episode(env, i)
        obs_buf[lo + i] = flatten_obs(obs)
        nact_buf[lo + i] = len(env._action_table)
        encode_action_table(env._action_table, acts_buf[lo + i])
    obs_ev.set()

    while True:
        act_ev.wait()
        act_ev.clear()
        for i, env in enumerate(envs):
            gi = lo + i
            obs, r, term, trunc, info = env.step(int(act_buf[gi]))
            done = term or trunc
            if shaping_coef:
                # F = gamma*PHI(s') - PHI(s), with PHI(terminal) = 0. Must be
                # read before reset, or the potential of the *next* episode's
                # opening state leaks into this episode's final reward.
                r += shapers[i].step(
                    None if done else env._inner._adapter.raw_state, done)
            if done:
                # Injected episodes are reported separately: they start past
                # ante 1, so mixing them into the headline would inflate it.
                stats_q.put(
                    (
                        info.get("balatro/ante_reached", 1),
                        info.get("balatro/rounds_beaten", 0),
                        bool(info.get("balatro/won", False)),
                        was_injected[i],
                    )
                )
                obs = start_episode(env, i)
            obs_buf[gi] = flatten_obs(obs)
            nact_buf[gi] = len(env._action_table)
            encode_action_table(env._action_table, acts_buf[gi])
            rew_buf[gi] = r
            done_buf[gi] = 1.0 if done else 0.0
        obs_ev.set()


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Pointer-policy PPO on Balatro")
    parser.add_argument("--total-steps", type=int, default=200_000_000)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--envs-per-worker", type=int, default=12)
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--minibatch", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--shaping-coef", type=float, default=1.0,
                        help="scale of the potential-based shaping term "
                             "(money/jokers/hand levels); 0 disables it")
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=3_000)
    parser.add_argument("--log-dir", type=str, default="runs/pointer_v3")
    parser.add_argument("--checkpoint-every", type=int, default=2_000_000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-center-emb", action="store_true",
                        help="disable joker/shop identity embeddings "
                             "(reproduces the pre-embedding baseline)")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--ent-coef-final", type=float, default=None,
                        help="linearly anneal --ent-coef to this by the end")
    parser.add_argument("--lr-final", type=float, default=None,
                        help="linearly anneal --lr to this by the end")
    parser.add_argument("--curriculum-pool", type=str, default=None,
                        help="pickle of mid-game states from "
                             "scripts/gen_curriculum.py")
    parser.add_argument("--curriculum-frac", type=float, default=0.0,
                        help="fraction of episodes restarted from the pool; "
                             "the rest start fresh at ante 1, and the two are "
                             "reported separately")
    parser.add_argument("--compile", choices=("none", "default", "cudagraph"),
                        default="none",
                        help="torch.compile the policy (bucketed action dim); "
                             "cudagraph also captures rollout forwards")
    args = parser.parse_args()

    from torch.utils.tensorboard import SummaryWriter

    from jackdaw.env.gymnasium_wrapper import MAX_ACTIONS

    from policy_v3 import OBS_DIM, PointerPolicy, bucket_actions, masked_dist

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    n_env = args.workers * args.envs_per_worker
    A = MAX_ACTIONS
    T = args.rollout
    log_path = Path(args.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_path))

    specs = {
        "obs": n_env * OBS_DIM * 4,
        "nact": n_env * 4,
        "act": n_env * 8,
        "rew": n_env * 4,
        "done": n_env * 4,
        "acts": n_env * A * 3 * 2,
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
    acts_buf = np.ndarray((n_env, A, 3), dtype=np.int16, buffer=shms["acts"].buf)
    acts_buf[:] = 0

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
                  act_evs[w], obs_evs[w], stats_q, args.max_steps,
                  args.gamma, args.shaping_coef, args.workers,
                  args.curriculum_pool, args.curriculum_frac),
            daemon=True,
        )
        p.start()
        procs.append(p)

    groups: list[list[int]] = [
        list(range(g, args.workers, args.groups)) for g in range(args.groups)
    ]
    group_idx_np = [
        np.concatenate([np.arange(w * args.envs_per_worker,
                                  (w + 1) * args.envs_per_worker) for w in g])
        for g in groups
    ]
    group_idx = [torch.as_tensor(i, device=device) for i in group_idx_np]

    net_cfg = {"d_model": args.d_model, "n_heads": args.n_heads,
               "n_layers": args.n_layers, "center_emb": not args.no_center_emb}
    policy = PointerPolicy(**net_cfg).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5, fused=True)
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        # Checkpoints written before net_cfg existed are all the d=128/2-layer
        # default, so a missing key is not ambiguous.
        ck_cfg = ckpt.get("net_cfg", {"d_model": 128, "n_heads": 4, "n_layers": 2})
        if ck_cfg != net_cfg:
            raise SystemExit(
                f"checkpoint architecture {ck_cfg} != requested {net_cfg}; "
                "resume needs matching --d-model/--n-heads/--n-layers")
        policy.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        global_step = ckpt["step"]
        print(f"resumed from {args.resume} at step {global_step:,}")

    # Rollout forwards are latency-bound: measured 1.82 ms/call eager, 0.85 ms
    # compiled, 0.46 ms with cuda graphs (d=128 net, batch 120, a_eff 128), so
    # the win is fewer kernel launches rather than more FLOPs. Bucketing the
    # action dim keeps shapes static enough for the compiled variants to stay
    # cached. Rollout and update get separate handles: graph capture is only
    # safe for the inference-mode rollout, the update keeps a normal compile.
    # Rollout and update each specialize on ~7 action-width buckets, and dynamo
    # caches per code object, so the two paths together blow past the default
    # limit of 8 and recompile on every call (update time doubled before this).
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 256
    if args.compile == "cudagraph":
        policy_roll = torch.compile(policy, dynamic=False, mode="reduce-overhead")
        policy_upd = torch.compile(policy, dynamic=False)
    elif args.compile == "default":
        policy_roll = policy_upd = torch.compile(policy, dynamic=False)
    else:
        policy_roll = policy_upd = policy

    obs_s = torch.zeros((T, n_env, OBS_DIM), device=device)
    acts_s = torch.zeros((T, n_env, A, 3), dtype=torch.int16, device=device)
    nact_s = torch.zeros((T, n_env), dtype=torch.long, device=device)
    act_s = torch.zeros((T, n_env), dtype=torch.long, device=device)
    logp_s = torch.zeros((T, n_env), device=device)
    val_s = torch.zeros((T, n_env), device=device)
    rew_s = torch.zeros((T, n_env), device=device)
    done_s = torch.zeros((T, n_env), device=device)

    for ev in obs_evs:
        ev.wait()
        ev.clear()
    last_obs = torch.as_tensor(obs_buf.copy(), device=device)
    last_acts = torch.as_tensor(acts_buf.copy(), device=device)
    last_n = torch.as_tensor(nact_buf.astype(np.int64), device=device)

    # Fresh-start episodes are the real objective; injected ones begin past
    # ante 1 and are tracked only to see whether the curriculum is being used
    # and whether skill there transfers back to fresh games.
    ep_antes: list[int] = []
    ep_rounds: list[int] = []
    ep_wins: list[bool] = []
    inj_antes: list[int] = []
    next_ckpt = (global_step // args.checkpoint_every + 1) * args.checkpoint_every
    print(f"{args.workers} workers x {args.envs_per_worker} envs = {n_env} envs, "
          f"{args.groups} groups, rollout {T} ({T * n_env:,} steps/iter), "
          f"action table {A}")
    print(f"policy d_model={args.d_model} heads={args.n_heads} "
          f"layers={args.n_layers} params={n_params:,} compile={args.compile}")

    start_step = global_step

    def _anneal(start: float, final: float | None) -> float:
        """Linear schedule over the remaining budget of this run."""
        if final is None or args.total_steps <= start_step:
            return start
        frac = (global_step - start_step) / (args.total_steps - start_step)
        return start + (final - start) * min(max(frac, 0.0), 1.0)

    while global_step < args.total_steps:
        t_start = time.time()
        lr_now = _anneal(args.lr, args.lr_final)
        ent_now = _anneal(args.ent_coef, args.ent_coef_final)
        for pg_group in opt.param_groups:
            pg_group["lr"] = lr_now

        with torch.inference_mode():
            for t in range(T):
                for gi, g in enumerate(groups):
                    idx, idx_np = group_idx[gi], group_idx_np[gi]
                    a_eff = bucket_actions(int(last_n[idx].max()))
                    o, ac_full = last_obs[idx], last_acts[idx]
                    with torch.autocast("cuda", torch.bfloat16):
                        logits, v = policy_roll(o, ac_full[:, :a_eff])
                    logits, v = logits.float(), v.float()
                    dist = masked_dist(logits, last_n[idx])
                    a = dist.sample()
                    obs_s[t, idx] = o
                    acts_s[t, idx] = ac_full
                    nact_s[t, idx] = last_n[idx]
                    act_s[t, idx] = a
                    logp_s[t, idx] = dist.log_prob(a)
                    val_s[t, idx] = v
                    act_buf[idx_np] = a.cpu().numpy()
                    for w in g:
                        act_evs[w].set()
                for gi, g in enumerate(groups):
                    for w in g:
                        obs_evs[w].wait()
                        obs_evs[w].clear()
                    idx, idx_np = group_idx[gi], group_idx_np[gi]
                    rew_s[t, idx] = torch.as_tensor(rew_buf[idx_np].copy(), device=device)
                    done_s[t, idx] = torch.as_tensor(done_buf[idx_np].copy(), device=device)
                    last_obs[idx] = torch.as_tensor(obs_buf[idx_np].copy(), device=device)
                    last_acts[idx] = torch.as_tensor(acts_buf[idx_np].copy(), device=device)
                    last_n[idx] = torch.as_tensor(
                        nact_buf[idx_np].astype(np.int64), device=device)

            with torch.autocast("cuda", torch.bfloat16):
                _, boot_v = policy_upd(
                    last_obs, last_acts[:, : bucket_actions(int(last_n.max()))])
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

        b_obs = obs_s.reshape(-1, OBS_DIM)
        b_acts = acts_s.reshape(-1, A, 3)
        b_nact = nact_s.reshape(-1)
        b_act = act_s.reshape(-1)
        b_logp = logp_s.reshape(-1)
        b_adv = adv.reshape(-1)
        # Normalize advantages over the whole batch, not per minibatch: since
        # minibatches are sorted by action-table width they each hold roughly
        # one game phase, and per-minibatch normalization would rescale that
        # phase's local noise to unit variance and erase cross-phase signal.
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        b_ret = ret.reshape(-1)
        batch = T * n_env
        # Even splits: a short trailing minibatch is a different shape, which
        # costs a torch.compile recompilation every iteration.
        n_mb = max(1, round(batch / args.minibatch))
        mb_size = batch // n_mb
        pg_l = vf_l = ent_l = kl = 0.0
        n_upd = 0
        for _ in range(args.epochs):
            # Minibatches are padded to the widest action table they contain,
            # so a uniform shuffle makes every one of them run at the global
            # max (~768) and the (B, A, d) activations dominate GPU memory.
            # Sorting by table width first keeps most minibatches narrow; the
            # random tie-break still reshuffles within a width each epoch.
            noise = torch.rand(batch, device=device)
            order = torch.argsort(b_nact.float() + noise)
            chunk_order = torch.randperm(n_mb, device=device).tolist()
            for c in chunk_order:
                mb = order[c * mb_size : (c + 1) * mb_size]
                a_eff = bucket_actions(int(b_nact[mb].max()))
                with torch.autocast("cuda", torch.bfloat16):
                    logits, v = policy_upd(b_obs[mb], b_acts[mb, :a_eff])
                logits, v = logits.float(), v.float()
                dist = masked_dist(logits, b_nact[mb])
                logp = dist.log_prob(b_act[mb])
                ratio = (logp - b_logp[mb]).exp()
                madv = b_adv[mb]
                pg = torch.max(
                    -madv * ratio,
                    -madv * ratio.clamp(1 - args.clip, 1 + args.clip),
                ).mean()
                vf = 0.5 * (v - b_ret[mb]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pg + args.vf_coef * vf - ent_now * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    pg_l += pg.item(); vf_l += vf.item(); ent_l += ent.item()
                    kl += (b_logp[mb] - logp).mean().item()
                n_upd += 1

        while not stats_q.empty():
            a_, r_, won, injected = stats_q.get_nowait()
            if injected:
                inj_antes.append(a_)
            else:
                ep_antes.append(a_); ep_rounds.append(r_); ep_wins.append(won)
        t_iter = time.time() - t_start
        sps = int(T * n_env / t_iter)
        writer.add_scalar("perf/sps", sps, global_step)
        writer.add_scalar("perf/collect_s", t_collect, global_step)
        writer.add_scalar("perf/update_s", t_iter - t_collect, global_step)
        writer.add_scalar("perf/gpu_peak_gb",
                          torch.cuda.max_memory_allocated() / 2**30, global_step)
        writer.add_scalar("loss/policy", pg_l / n_upd, global_step)
        writer.add_scalar("loss/value", vf_l / n_upd, global_step)
        writer.add_scalar("loss/entropy", ent_l / n_upd, global_step)
        writer.add_scalar("loss/approx_kl", kl / n_upd, global_step)
        writer.add_scalar("hp/lr", lr_now, global_step)
        writer.add_scalar("hp/ent_coef", ent_now, global_step)
        line = (f"step {global_step:>12,}  sps {sps:>6,}  "
                f"c/u {t_collect:.1f}/{t_iter - t_collect:.1f}s  "
                f"gpu {torch.cuda.max_memory_allocated() / 2**30:.1f}G")
        if ep_antes:
            writer.add_scalar("balatro/mean_ante", np.mean(ep_antes), global_step)
            writer.add_scalar("balatro/max_ante", np.max(ep_antes), global_step)
            writer.add_scalar("balatro/mean_rounds", np.mean(ep_rounds), global_step)
            writer.add_scalar("balatro/win_rate", np.mean(ep_wins), global_step)
            line += (f"  ante {np.mean(ep_antes):.2f} (max {np.max(ep_antes)})"
                     f"  win {100 * np.mean(ep_wins):.1f}%"
                     f"  eps {len(ep_antes)}")
            ep_antes.clear(); ep_rounds.clear(); ep_wins.clear()
        if inj_antes:
            writer.add_scalar("balatro/mean_ante_injected",
                              np.mean(inj_antes), global_step)
            line += f"  [inj {np.mean(inj_antes):.2f} x{len(inj_antes)}]"
            inj_antes.clear()
        print(line, flush=True)

        if global_step >= next_ckpt:
            ck = {"model": policy.state_dict(), "opt": opt.state_dict(),
                  "step": global_step, "net_cfg": net_cfg}
            torch.save(ck, log_path / f"ckpt_{global_step}.pt")
            torch.save(ck, log_path / "latest.pt")
            next_ckpt += args.checkpoint_every

    torch.save({"model": policy.state_dict(), "opt": opt.state_dict(),
                "step": global_step, "net_cfg": net_cfg}, log_path / "final.pt")
    print(f"done — saved {log_path / 'final.pt'}")


if __name__ == "__main__":
    main()
