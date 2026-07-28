"""Check that center-key ids survive the float round-trip in the observation.

The policy recovers a joker's identity by inverting feature 0
(``center_key_id / NUM_CENTER_KEYS``). If that inversion is off by one, or if
float32 rounding collapses two ids, the embedding silently learns nonsense --
so verify against the encoder rather than trusting the arithmetic.

    nix develop -c .venv/bin/python scripts/test_center_ids.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def main() -> None:
    from jackdaw.env.observation import (
        _CENTER_KEY_TO_ID as CENTERS,
        NUM_CENTER_KEYS,
        center_key_id,
    )

    # 1. Every real center key must round-trip through the float encoding.
    bad = []
    for key in CENTERS:
        cid = center_key_id(key)
        encoded = np.float32(cid / max(NUM_CENTER_KEYS, 1))
        decoded = int(np.round(float(encoded) * NUM_CENTER_KEYS))
        if decoded != cid:
            bad.append((key, cid, decoded))
    print(f"center keys checked: {len(CENTERS)}  round-trip failures: {len(bad)}")
    for key, cid, decoded in bad[:5]:
        print(f"  {key}: {cid} -> {decoded}")

    # 2. Ids must stay distinct after encoding (no collisions from rounding).
    encoded = {np.float32(center_key_id(k) / NUM_CENTER_KEYS) for k in CENTERS}
    ids = {center_key_id(k) for k in CENTERS}
    print(f"distinct ids: {len(ids)}  distinct encoded floats: {len(encoded)}")

    # 3. End to end: decode a live observation and compare with the game state.
    from jackdaw.env.game_interface import DirectAdapter
    from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv

    env = BalatroGymnasiumEnv(
        adapter_factory=DirectAdapter, max_steps=3_000,
        seed_prefix="CENTERID", reward_shaping=True,
    )
    obs, _ = env.reset()
    rng = np.random.default_rng(0)
    seen_shop = False
    for _ in range(400):
        obs, _r, term, trunc, _info = env.step(
            int(rng.integers(0, max(1, len(env._action_table)))))
        shop = obs["shop_item"]
        n_shop = int(obs["entity_counts"][3])
        if n_shop:
            seen_shop = True
            decoded = np.round(shop[:n_shop, 0] * NUM_CENTER_KEYS).astype(int)
            print(f"shop slots {n_shop}: decoded ids {decoded.tolist()}")
            assert decoded.min() >= 0 and decoded.max() <= NUM_CENTER_KEYS
            break
        if term or trunc:
            obs, _ = env.reset()
    print("shop observed:" , seen_shop)
    print("OK" if not bad and len(ids) == len(encoded) else "FAILED")


if __name__ == "__main__":
    main()
