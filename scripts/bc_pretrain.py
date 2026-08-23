"""Behavior-cloning pretrain for the pointer policy.

Supervised phase of the two-phase pipeline: imitate heuristic card
selections so the network starts with poker hand formation and basic
play competence, before RL fine-tuning adapts it to joker context.

    nix develop -c .venv/bin/python scripts/bc_pretrain.py \
        --data runs/bc/demo.npz --out runs/bc/bc_init.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from policy_v3 import PointerPolicy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/bc/demo.npz")
    ap.add_argument("--out", default="runs/bc/bc_init.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    data = np.load(args.data)
    obs = torch.as_tensor(data["obs"], dtype=torch.float32)
    acts = torch.as_tensor(data["acts"], dtype=torch.long)
    nact = torch.as_tensor(data["nact"], dtype=torch.long)
    label = torch.as_tensor(data["label"], dtype=torch.long)
    n = len(label)
    print(f"samples={n} obs={obs.shape}")

    net_cfg = {"d_model": 128, "n_heads": 4, "n_layers": 2,
               "center_emb": True}
    policy = PointerPolicy(**net_cfg)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    idx_all = np.arange(n)
    for epoch in range(args.epochs):
        np.random.shuffle(idx_all)
        total_loss, correct, seen = 0.0, 0, 0
        for lo in range(0, n, args.batch):
            idx = idx_all[lo:lo + args.batch]
            x, a, na, y = (obs[idx], acts[idx], nact[idx], label[idx])
            logits, _ = policy(x, a)
            loss = F.cross_entropy(logits, y, reduction="none")
            valid = torch.arange(logits.shape[-1])[None] < na[:, None]
            masked_logits = logits.masked_fill(~valid, -1e9)
            loss = F.cross_entropy(masked_logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss) * len(idx)
            correct += int((masked_logits.argmax(-1) == y).sum())
            seen += len(idx)
        print(f"epoch {epoch + 1}: loss={total_loss / max(seen,1):.4f} "
              f"top-1 match={100 * correct / max(seen,1):.1f}%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": policy.state_dict(),
                "opt": opt.state_dict(),
                "step": 0,
                "net_cfg": net_cfg}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
