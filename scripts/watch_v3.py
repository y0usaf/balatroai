"""Watch a v3 pointer-policy checkpoint play REAL rendered Balatro.

The model needs full engine observations, which the live game can't
provide — so it plays on the in-process jackdaw sim while every action
is mirrored to a live balatrobot game started with the same seed.
jackdaw is validated 1:1 against live Balatro, so the two stay in
lockstep; light state comparison runs each step and any divergence is
reported.

Usage::

    # with a balatrobot server already running:
    uv run python -u scripts/watch_v3.py runs/pointer_v8/latest.pt

    # or let it spawn the isolated game copy itself:
    uv run python -u scripts/watch_v3.py runs/pointer_v8/latest.pt --launch
"""

from __future__ import annotations

import argparse
import secrets
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jackdaw.bridge.balatrobot_adapter import action_to_rpc
from jackdaw.env.action_space import factored_to_engine_action
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.gymnasium_wrapper import MAX_ACTIONS, BalatroGymnasiumEnv

from policy_v3 import PointerPolicy, encode_action_table, flatten_obs, masked_dist

ACTION_NAMES = [
    "play", "discard", "select blind", "skip blind", "cash out", "reroll",
    "next round", "skip pack", "buy", "sell joker", "sell consumable",
    "use consumable", "redeem voucher", "open booster", "pick pack card",
    "swap jokers L", "swap jokers R", "swap hand L", "swap hand R",
    "sort rank", "sort suit",
]


def describe(fa, gs) -> str:
    name = ACTION_NAMES[fa.action_type]
    bits = []
    if fa.card_target:
        hand = gs.get("hand", [])
        labels = []
        for i in fa.card_target:
            c = hand[i] if i < len(hand) else None
            b = getattr(c, "base", None)
            labels.append(f"{getattr(b, 'value', '?')}{str(getattr(b, 'suit', '?'))[:1]}")
        bits.append(" ".join(labels))
    if fa.entity_target is not None:
        bits.append(f"#{fa.entity_target}")
    return f"{name}" + (f" [{', '.join(bits)}]" if bits else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror a v3 checkpoint onto live Balatro")
    parser.add_argument("model", nargs="?", default="runs/pointer_v8/latest.pt")
    parser.add_argument("--seed", type=int, default=None,
                        help="game seed (default: random)")
    parser.add_argument("--port", type=int, default=12346)
    parser.add_argument("--gamespeed", type=int, default=2)
    parser.add_argument("--launch", action="store_true",
                        help="spawn `uvx balatrobot serve` (isolated copy) first")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="extra pause between actions (s)")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    game_seed = (
        args.seed if args.seed is not None else secrets.randbelow(1_000_000_000)
    )

    from balatroai.client import Client, RPCError
    live_client = Client(port=args.port)

    if args.launch:
        from balatroai import gamedir
        from balatroai.cli import launch_server
        game = gamedir.default_game_dir()
        print(f"launching balatrobot (rendered, {args.gamespeed}x) …")
        launch_server(args.port, headless=False,
                      game=game if gamedir.is_set_up(game) else None,
                      gamespeed=args.gamespeed)
    if not live_client.wait_ready(timeout=180 if args.launch else 5):
        sys.exit(f"no balatrobot server on port {args.port} "
                 "(start one or pass --launch)")

    ckpt = torch.load(args.model, map_location="cpu")
    # Pre-net_cfg checkpoints are all the d=128/2-layer default.
    policy = PointerPolicy(**ckpt.get(
        "net_cfg", {"d_model": 128, "n_heads": 4, "n_layers": 2}))
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    print(f"model: {args.model} @ step {ckpt.get('step', '?'):,}")

    live = live_client
    seed = str(game_seed)
    print(f"seed {seed} · deck RED · stake WHITE\n")

    env = BalatroGymnasiumEnv(
        adapter_factory=DirectAdapter, max_steps=3_000,
        seed_prefix="WATCH", reward_shaping=False,
    )
    obs, info = env.reset(seed=game_seed)
    try:
        live.call("menu", {})
    except RPCError:
        pass
    live_state = live.call("start", {"deck": "RED", "stake": "WHITE", "seed": seed})

    struct = np.zeros((MAX_ACTIONS, 3), dtype=np.int16)
    step = 0
    mirrored = True
    with torch.inference_mode():
        done = False
        while not done:
            step += 1
            gs = env._inner._adapter.raw_state
            table = env._action_table
            n = len(table)
            x = torch.as_tensor(flatten_obs(obs)).unsqueeze(0)
            encode_action_table(table, struct)
            acts = torch.as_tensor(struct[:n]).unsqueeze(0)
            logits, _ = policy(x, acts)
            if args.stochastic:
                a = masked_dist(logits, torch.as_tensor([n])).sample().item()
            else:
                a = logits[0, :n].argmax().item()
            fa = table[a]

            engine_action = factored_to_engine_action(fa, gs)
            note = describe(fa, gs)

            # mirror to live first (it's the slow one), then sim
            if mirrored:
                try:
                    rpc = action_to_rpc(engine_action, gs)
                    live_state = live.call(rpc["method"], rpc["params"])
                except RPCError as e:
                    print(f"\n!! live rejected {note}: {e} — sim continues unmirrored")
                    mirrored = False

            obs, _, term, trunc, info = env.step(int(a))
            done = term or trunc

            sim_gs = env._inner._adapter.raw_state
            ante = sim_gs.get("round_resets", {}).get("ante", "?")
            money = sim_gs.get("dollars", "?")
            line = f"[{step:4d}] ante {ante} ${money}  {note}"
            if mirrored:
                lm = live_state.get("money")
                la = live_state.get("ante_num")
                if lm is not None and lm != money or la is not None and la != ante:
                    line += f"  ⚠ DIVERGED (live: ante {la} ${lm})"
            print(line)
            if args.delay:
                time.sleep(args.delay)

    won = info.get("balatro/won", False)
    print(f"\n{'WON' if won else 'game over'} — ante {info.get('balatro/ante_reached')}"
          f" · rounds {info.get('balatro/rounds_beaten')}")


if __name__ == "__main__":
    main()
