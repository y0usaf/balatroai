"""Build a curriculum pool of mid-game states from heuristic-bot games.

A from-scratch policy dies in the first blind, so it never reaches a shop and
never experiences a joker firing -- there is no gradient for an event you never
see. This records real mid-game positions (ante >= 2, with money and jokers on
board) that training can restart from, so those situations become common
instead of unreachable.

Two properties keep this from becoming seed memorization:

* **breadth** -- states come from many distinct seeds, and each game
  contributes only a few snapshots, so no single run dominates the pool;
* **fresh randomness on use** -- the trainer reseeds the PRNG when it injects
  a state, so one snapshot yields a different future every time it is used
  (different draws, different shop). The board is reused; the game is not.

The heuristic decides only *which situations* get recorded, never which action
is correct, so its hand-maximizing bias does not transfer to the policy.

    nix develop -c .venv/bin/python scripts/gen_curriculum.py --games 300
"""

from __future__ import annotations

import argparse
import copy
import pickle
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Record mid-game states for curriculum")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--bot", default="heuristic")
    ap.add_argument("--min-ante", type=int, default=2,
                    help="only record states at or past this ante")
    ap.add_argument("--per-game", type=int, default=4,
                    help="cap snapshots per game, so no seed dominates the pool")
    ap.add_argument("--out", type=Path, default=Path("runs/curriculum/pool.pkl"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from balatroai.bots import get_bot
    from balatroai.cli import random_seed
    from balatroai.runner import Runner
    from balatroai.sim import SimClient

    rng = random.Random(args.seed)
    pool: list[dict] = []
    antes: dict[int, int] = {}
    t0 = time.time()

    for g in range(args.games):
        client = SimClient()
        bot = get_bot(args.bot)
        taken: list[dict] = []

        def emit(state: dict, _action, _error, _prev=None) -> None:
            ante = state.get("ante_num", 0) or 0
            if ante < args.min_ante:
                return
            gs = client._backend._gs
            if gs is None:
                return
            # Reservoir-style: keep at most --per-game per run, but let later
            # states replace earlier ones so deep positions are represented.
            snap = {"gs": copy.deepcopy(gs), "ante": ante,
                    "round": state.get("round_num", 0)}
            if len(taken) < args.per_game:
                taken.append(snap)
            elif rng.random() < 0.3:
                taken[rng.randrange(len(taken))] = snap

        runner = Runner(client, bot, emit=emit)
        # A seedless start replays one fixed game, which would make the whole
        # pool a single seed -- exactly the memorization risk this must avoid.
        result = runner.play_game(seed=random_seed())
        pool.extend(taken)
        for s in taken:
            antes[s["ante"]] = antes.get(s["ante"], 0) + 1

        if (g + 1) % 25 == 0:
            print(f"game {g + 1}/{args.games}  pool {len(pool):,}  "
                  f"last ante {result.ante}  {time.time() - t0:.0f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(pool, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = args.out.stat().st_size / 2**20
    print(f"\nwrote {len(pool):,} states to {args.out} ({size_mb:.1f} MB)")
    print("ante distribution:", dict(sorted(antes.items())))


if __name__ == "__main__":
    main()
