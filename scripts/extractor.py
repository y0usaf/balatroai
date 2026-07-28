"""Entity-attention features extractor for the jackdaw Balatro env.

SB3's default ``MultiInputPolicy`` flattens the padded entity arrays
(hand cards, jokers, shop items...), so "card in slot 3" and "card in
slot 4" hit different weights and card/joker interactions must be
memorized per-position.  This extractor instead:

- embeds each entity row with a type-specific projection + type embedding
- prepends a global token built from the global feature vector
- runs a small Transformer encoder over the token sequence with
  key-padding masks derived from ``entity_counts``
- returns [global token ‖ masked mean of entity tokens] → linear head

giving permutation invariance over slots and explicit entity-entity
interaction modeling (joker × hand-card synergies, shop trade-offs).
"""

from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from jackdaw.env.balatro_spec import balatro_game_spec


class EntityAttentionExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        features_dim: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim)

        spec = balatro_game_spec()
        # (name, max_count, feature_dim) — order must match entity_counts
        self._entities: list[tuple[str, int, int]] = [
            (et.name, et.max_count, et.feature_dim) for et in spec.entity_types
        ]
        global_dim = observation_space["global"].shape[0]

        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, d_model), nn.LayerNorm(d_model)
        )
        self.entity_proj = nn.ModuleDict(
            {name: nn.Linear(feat, d_model) for name, _, feat in self._entities}
        )
        self.type_emb = nn.Embedding(len(self._entities), d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, features_dim), nn.ReLU()
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        batch = obs["global"].shape[0]
        device = obs["global"].device
        counts = obs["entity_counts"]  # (B, n_types) float

        tokens = [self.global_proj(obs["global"]).unsqueeze(1)]  # (B, 1, d)
        valid = [torch.ones(batch, 1, dtype=torch.bool, device=device)]
        for i, (name, max_count, _) in enumerate(self._entities):
            emb = self.entity_proj[name](obs[name])  # (B, max_count, d)
            emb = emb + self.type_emb.weight[i]
            tokens.append(emb)
            slots = torch.arange(max_count, device=device).unsqueeze(0)  # (1, max_count)
            valid.append(slots < counts[:, i].unsqueeze(1))

        seq = torch.cat(tokens, dim=1)          # (B, 1+T, d)
        mask = torch.cat(valid, dim=1)          # (B, 1+T) True = real
        out = self.encoder(seq, src_key_padding_mask=~mask)

        g = out[:, 0]                            # global token
        ent = out[:, 1:]
        ent_mask = mask[:, 1:].unsqueeze(-1).float()
        pooled = (ent * ent_mask).sum(1) / ent_mask.sum(1).clamp(min=1.0)
        return self.head(torch.cat([g, pooled], dim=-1))
