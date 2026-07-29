"""Pointer policy: actions scored by *content*, not table slot.

v2's flat Discrete head had to learn "slot i is good in state s" while the
wrapper's enumeration (and its random subsampling) shuffled what slot i
*meant* — deterministic play collapsed to ante 1.  Here each enumerated
action is embedded from its parts:

    emb = type_emb[action_type]
        + entity token (from the trunk) for the targeted joker/shop/pack item
        + mean of the hand-card tokens it selects

and scored against a query from the trunk state:  logit = q · key(emb) / √d.
Slot order becomes irrelevant; "play these 5 flush cards" has the same
representation wherever it lands in the table.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.gymnasium_wrapper import MAX_ACTIONS
from jackdaw.env.observation import NUM_CENTER_KEYS

from policy import ENTITIES, GLOBAL_DIM, OBS_DIM, flatten_obs, unflatten_obs  # noqa: F401 (re-export)

_SPEC = balatro_game_spec()
N_ACTION_TYPES = len(_SPEC.action_types)
# Jokers, consumables and shop items are identified only by feature 0,
# center_key_id / NUM_CENTER_KEYS. As a scalar that is nearly useless: two
# consecutive ids are unrelated effects but look almost identical, so the net
# cannot learn what any specific joker does. Recover the id and embed it.
_KEYED_ENTITIES = {"joker", "consumable", "shop_item"}
# action type -> entity type index (-1 = no entity target)
_ETI = [at.entity_type_index for at in _SPEC.action_types]
# entity type index -> base offset in the token sequence (global token at 0)
_TOKEN_BASE: list[int] = []
_off = 1
for _name, _count, _feat in ENTITIES:
    _TOKEN_BASE.append(_off)
    _off += _count
N_TOKENS = _off
HAND_MAX = ENTITIES[0][1]  # hand_card is entity type 0


# Action tables are rebuilt every step and can hold ~768 rows, so this is hot:
# it was 101 us/step (22% of a worker's CPU) when written as per-cell numpy
# item assignment. Build plain Python rows, then hand numpy one array.
_BITMASK_CACHE: dict[tuple[int, ...], int] = {}


def _bitmask(cards) -> int:
    key = tuple(cards)
    bm = _BITMASK_CACHE.get(key)
    if bm is None:
        bm = 0
        for c in key:
            bm |= 1 << c
        _BITMASK_CACHE[key] = bm
    return bm


def encode_action_table(table, out: np.ndarray) -> None:
    """Encode a wrapper action table into an (A, 3) int16 array in-place.

    Columns: [action_type, entity_target (-1 = none), card bitmask].
    Only ``len(table)`` rows are written; callers mask by n_actions.
    """
    n = len(table)
    if n == 0:
        return
    rows = [
        (
            fa.action_type,
            -1 if fa.entity_target is None else fa.entity_target,
            _bitmask(fa.card_target) if fa.card_target else 0,
        )
        for fa in table
    ]
    out[:n] = np.array(rows, dtype=np.int16)


# Padding buckets for the action dimension. torch.compile / cuda graphs need
# static shapes; without bucketing every distinct table width triggers a
# recompile, with it we pay at most len(A_BUCKETS) compilations.
A_BUCKETS: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 768)


def bucket_actions(n: int) -> int:
    """Smallest bucket >= n (the largest bucket for anything oversized)."""
    for b in A_BUCKETS:
        if n <= b:
            return b
    return A_BUCKETS[-1]


class PointerPolicy(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        center_emb: bool = True,
    ) -> None:
        super().__init__()
        self.d = d_model
        self.global_proj = nn.Sequential(
            nn.Linear(GLOBAL_DIM, d_model), nn.LayerNorm(d_model)
        )
        self.entity_proj = nn.ModuleDict(
            {name: nn.Linear(feat, d_model) for name, _, feat in ENTITIES}
        )
        self.ent_type_emb = nn.Embedding(len(ENTITIES), d_model)
        # Shared across joker/consumable/shop_item: the same center key means
        # the same object whether it sits in the shop or in your joker slots,
        # and ent_type_emb already tells the net which of the two it is.
        # Switchable so ablations can reproduce the pre-embedding baseline.
        self.center_emb = (
            nn.Embedding(NUM_CENTER_KEYS + 1, d_model) if center_emb else None
        )
        if self.center_emb is not None:
            # nn.Embedding defaults to N(0, 1), which is ~4x the scale of the
            # entity_proj output it is added to: every joker token would start
            # buried under a random vector, and ids that are rarely drawn would
            # keep theirs. Ablation measured that cost as 1.70 -> 1.27 mean
            # ante. Start near zero and let the net earn the separation.
            nn.init.normal_(self.center_emb.weight, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.act_type_emb = nn.Embedding(N_ACTION_TYPES, d_model)
        self.key_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.query = nn.Linear(2 * d_model, d_model)
        self.critic = nn.Sequential(
            nn.Linear(2 * d_model, 256), nn.ReLU(), nn.Linear(256, 1)
        )

        self.register_buffer(
            "eti", torch.tensor(_ETI, dtype=torch.long), persistent=False)
        self.register_buffer(
            "token_base", torch.tensor(_TOKEN_BASE, dtype=torch.long), persistent=False)
        self.register_buffer(
            "bit_idx", torch.arange(HAND_MAX, dtype=torch.int16), persistent=False)

    def forward(
        self, x_flat: torch.Tensor, acts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x_flat: (B, OBS_DIM) float; acts: (B, A, 3) int16/long."""
        obs = unflatten_obs(x_flat)
        batch = x_flat.shape[0]
        device = x_flat.device
        counts = obs["entity_counts"]

        tokens = [self.global_proj(obs["global"]).unsqueeze(1)]
        valid = [torch.ones(batch, 1, dtype=torch.bool, device=device)]
        for i, (name, max_count, _) in enumerate(ENTITIES):
            emb = self.entity_proj[name](obs[name]) + self.ent_type_emb.weight[i]
            if self.center_emb is not None and name in _KEYED_ENTITIES:
                # Feature 0 is center_key_id / NUM_CENTER_KEYS; invert it.
                # Empty slots decode to id 0, which the padding mask drops.
                ids = (obs[name][..., 0] * NUM_CENTER_KEYS).round().long()
                emb = emb + self.center_emb(ids.clamp(0, NUM_CENTER_KEYS))
            tokens.append(emb)
            slots = torch.arange(max_count, device=device).unsqueeze(0)
            valid.append(slots < counts[:, i].unsqueeze(1))
        seq = torch.cat(tokens, dim=1)
        vmask = torch.cat(valid, dim=1)
        out = self.encoder(seq, src_key_padding_mask=~vmask)

        g = out[:, 0]
        ent_mask = vmask[:, 1:].unsqueeze(-1).float()
        pooled = (out[:, 1:] * ent_mask).sum(1) / ent_mask.sum(1).clamp(min=1.0)
        feat = torch.cat([g, pooled], dim=-1)
        value = self.critic(feat).squeeze(-1)

        # ---- action embeddings ----
        acts = acts.long()
        at = acts[..., 0].clamp(0, N_ACTION_TYPES - 1)          # (B, A)
        ent = acts[..., 1]                                       # (B, A)
        bm = acts[..., 2]                                        # (B, A)

        emb = self.act_type_emb(at)
        eti = self.eti[at]                                       # (B, A)
        has_ent = (eti >= 0) & (ent >= 0)
        tok_idx = (self.token_base[eti.clamp(min=0)] + ent.clamp(min=0)).clamp(
            0, N_TOKENS - 1)
        ent_tok = out.gather(1, tok_idx.unsqueeze(-1).expand(-1, -1, self.d))
        emb = emb + ent_tok * has_ent.unsqueeze(-1)

        bits = ((bm.unsqueeze(-1) >> self.bit_idx.long()) & 1).float()  # (B, A, 8)
        hand_toks = out[:, 1 : 1 + HAND_MAX]                     # (B, 8, d)
        card_sum = bits @ hand_toks                              # (B, A, d)
        emb = emb + card_sum / bits.sum(-1, keepdim=True).clamp(min=1.0)

        keys = self.key_mlp(emb)                                 # (B, A, d)
        q = self.query(feat).unsqueeze(-1)                       # (B, d, 1)
        logits = (keys @ q).squeeze(-1) / math.sqrt(self.d)      # (B, A)
        return logits, value


def masked_dist(
    logits: torch.Tensor, n_actions: torch.Tensor
) -> torch.distributions.Categorical:
    idx = torch.arange(logits.shape[-1], device=logits.device)
    mask = idx.unsqueeze(0) < n_actions.unsqueeze(-1)
    return torch.distributions.Categorical(logits=logits.masked_fill(~mask, -1e9))
