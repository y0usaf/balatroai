"""Shared policy net + flat observation layout for the fast trainer.

The jackdaw gymnasium env returns dict observations; for shared-memory
transport we flatten to a single float32 vector per env:

    [global | hand_card | joker | consumable | shop_item | pack_card | counts]

and the legal-action mask is a *prefix* mask (the wrapper enumerates the
action table densely), so workers ship a single int ``n_actions``.
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.gymnasium_wrapper import MAX_ACTIONS

from extractor import EntityAttentionExtractor

_SPEC = balatro_game_spec()
ENTITIES: list[tuple[str, int, int]] = [
    (et.name, et.max_count, et.feature_dim) for et in _SPEC.entity_types
]
GLOBAL_DIM: int = _SPEC.global_feature_dim
OBS_DIM: int = GLOBAL_DIM + sum(c * f for _, c, f in ENTITIES) + len(ENTITIES)


def obs_space() -> spaces.Dict:
    d: dict[str, spaces.Space] = {
        "global": spaces.Box(-np.inf, np.inf, shape=(GLOBAL_DIM,), dtype=np.float32),
    }
    for name, count, feat in ENTITIES:
        d[name] = spaces.Box(-np.inf, np.inf, shape=(count, feat), dtype=np.float32)
    d["entity_counts"] = spaces.Box(
        0, np.inf, shape=(len(ENTITIES),), dtype=np.float32
    )
    return spaces.Dict(d)


def flatten_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    parts = [obs["global"].ravel()]
    parts += [obs[name].ravel() for name, _, _ in ENTITIES]
    parts.append(obs["entity_counts"].ravel())
    return np.concatenate(parts, dtype=np.float32)


def unflatten_obs(x: torch.Tensor) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {"global": x[:, :GLOBAL_DIM]}
    off = GLOBAL_DIM
    for name, count, feat in ENTITIES:
        out[name] = x[:, off : off + count * feat].reshape(-1, count, feat)
        off += count * feat
    out["entity_counts"] = x[:, off : off + len(ENTITIES)]
    return out


def _layer_init(layer: nn.Linear, std: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class PolicyNet(nn.Module):
    """Entity-attention trunk + actor/critic heads over the flat action table."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        features_dim: int = 256,
    ) -> None:
        super().__init__()
        self.extractor = EntityAttentionExtractor(
            obs_space(),
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            features_dim=features_dim,
        )
        self.actor = _layer_init(nn.Linear(features_dim, MAX_ACTIONS), std=0.01)
        self.critic = _layer_init(nn.Linear(features_dim, 1), std=1.0)

    def forward(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.extractor(unflatten_obs(x_flat))
        return self.actor(f), self.critic(f).squeeze(-1)


def masked_dist(
    logits: torch.Tensor, n_actions: torch.Tensor
) -> torch.distributions.Categorical:
    """Categorical over the first ``n_actions`` slots of each row."""
    idx = torch.arange(logits.shape[-1], device=logits.device)
    mask = idx.unsqueeze(0) < n_actions.unsqueeze(-1)
    return torch.distributions.Categorical(logits=logits.masked_fill(~mask, -1e9))
