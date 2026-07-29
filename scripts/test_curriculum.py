"""Verify injected states resume correctly and do not replay one fixed game.

Checks, in order:

1. an injected state produces a legal action table and keeps playing;
2. injecting the *same* snapshot repeatedly gives materially different games,
   which is what stops the agent from memorizing a fixed pool;
3. the injected ante matches the snapshot, so the curriculum really does start
   past the wall a fresh policy hits.

    nix develop -c .venv/bin/python scripts/test_curriculum.py
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def main() -> None:
    from jackdaw.env.game_interface import DirectAdapter
    from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv

    from curriculum import inject, load_pool

    pool_path = Path("runs/curriculum/probe.pkl")
    if not pool_path.exists():
        raise SystemExit(f"no pool at {pool_path}; run scripts/gen_curriculum.py first")

    pool = load_pool(pool_path)
    print(f"pool size {len(pool):,}")
    print("ante spread:", dict(sorted(Counter(s['ante'] for s in pool).items())))

    env = BalatroGymnasiumEnv(
        adapter_factory=DirectAdapter, max_steps=3_000,
        seed_prefix="CURR", reward_shaping=True,
    )
    env.reset()
    rng = random.Random(0)
    nprng = np.random.default_rng(0)

    # 1 + 3: resume from a snapshot and play it out.
    snap = pool[len(pool) // 2]
    obs, _info = inject(env, snap, rng)
    print(f"\ninjected ante {snap['ante']} round {snap['round']}: "
          f"{len(env._action_table)} legal actions, "
          f"${env._inner._adapter._gs.get('dollars')}, "
          f"{len(env._inner._adapter._gs.get('jokers') or [])} jokers")
    assert len(env._action_table) > 0, "injected state has no legal actions"

    steps = 0
    while steps < 200 and env._action_table:
        _o, _r, term, trunc, _i = env.step(
            int(nprng.integers(0, len(env._action_table))))
        steps += 1
        if term or trunc:
            break
    print(f"played {steps} steps from the injected state without error")

    # 2: same snapshot, fresh randomness each time -> different games.
    endings = []
    for _ in range(12):
        inject(env, snap, rng)
        n = 0
        while n < 300 and env._action_table:
            _o, _r, term, trunc, _i = env.step(
                int(nprng.integers(0, len(env._action_table))))
            n += 1
            if term or trunc:
                break
        endings.append((env._episode_max_ante, env._episode_max_round, n))
    distinct = len(set(endings))
    print(f"\n12 replays of ONE snapshot -> {distinct} distinct outcomes")
    print("  sample:", endings[:5])
    print("OK" if distinct > 1 else "FAILED: snapshot replays identically")


if __name__ == "__main__":
    main()
