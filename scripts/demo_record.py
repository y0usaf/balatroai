"""Record (obs, n_actions, expert_action) tuples for behavior cloning.

Drives BalatroGymnasiumEnv with HeuristicBot through heuristic_view.build_state,
mapping each heuristic Action to its arm index in the card-level action table
(exact set match, else best-overlap).  Saves flattened observations + labels
for supervised pretraining of the pointer policy.

    nix develop -c .venv/bin/python scripts/demo_record.py --games 400 --out runs/bc/demo.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jackdaw.env.action_space import ActionType
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.gymnasium_wrapper import MAX_ACTIONS, BalatroGymnasiumEnv

from heuristic_view import build_state
from policy_v3 import encode_action_table, flatten_obs


def find_arm(env, act):
    want_type = AT.get(act.method)
    if want_type is None:
        return None
    cards = set(act.params.get("cards") or [])
    ent = act.params.get("card", act.params.get("joker",
                  act.params.get("consumable")))
    best, best_ov = None, -1
    for i, fa in enumerate(env._action_table):
        if int(fa.action_type) != int(want_type):
            continue
        if act.method in ("play", "discard"):
            ov = len(set(fa.card_target or ()) & cards)
            if set(fa.card_target or ()) == cards:
                return i
            if ov > best_ov:
                best, best_ov = i, ov
        elif act.method in ("buy", "sell", "use"):
            if int(fa.entity_target or -2) == int(ent if ent is not None else -2):
                return i
        else:
            return i
    return best


AT = {"play": ActionType.PlayHand, "discard": ActionType.Discard,
      "buy": ActionType.BuyCard, "sell": ActionType.SellJoker,
      "use": ActionType.UseConsumable, "reroll": ActionType.Reroll,
      "select": ActionType.SelectBlind, "cash_out": ActionType.CashOut,
      "next_round": ActionType.NextRound, "pack": ActionType.OpenBooster}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--bot", default="heuristic")
    ap.add_argument("--out", type=str, default="runs/bc/demo.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from balatroai.bots import get_bot

    bot = get_bot(args.bot)
    obs_list, nact_list, label_list = [], [], []
    acts_list = []
    rng = np.random.default_rng(args.seed)

    for g in range(args.games):
        env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, max_steps=2000,
                                  seed_prefix=f"BC{g}", reward_shaping=True)
        obs, _ = env.reset()
        for t in range(1500):
            raw = env._inner._adapter.raw_state
            act = bot.act(build_state(raw))
            label = find_arm(env, act)
            x = flatten_obs(obs)
            n_act = len(env._action_table)
            acts = np.zeros((1, MAX_ACTIONS, 3), dtype=np.int16)
            encode_action_table(env._action_table, acts[0])
            if label is not None:
                obs_list.append(x.astype(np.float32))
                nact_list.append(n_act)
                label_list.append(label)
                acts_list.append(acts[0].astype(np.int16))
            a = label if label is not None else \
                int(rng.integers(0, max(1, n_act)))
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        if (g + 1) % 50 == 0:
            print(f"game {g+1}/{args.games}: {len(label_list)} labeled steps",
                  flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        obs=np.array(obs_list, dtype=np.float32),
        nact=np.array(nact_list, dtype=np.int64),
        label=np.array(label_list, dtype=np.int64),
        acts=np.array(acts_list, dtype=np.int16),
    )
    print(f"wrote {len(label_list)} samples -> {out}")


if __name__ == "__main__":
    main()
