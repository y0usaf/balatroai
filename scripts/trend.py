"""Best-fit trend of avg_ante vs total_timesteps from an SB3 training log.

Usage: python scripts/trend.py [RUN_DIR]   (default: runs/masked_v2)

Walks the log, pairs each eval's avg_ante with the most recent
total_timesteps, then reports OLS slope/correlation, split-half slopes,
and a log-time projection.  With few points the fit is noise — see the
n<15 warning.
"""
import sys
from pathlib import Path

import numpy as np

run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/masked_v2")
pairs, pending, last_ts = [], [], None
for line in open(run_dir / "train.log"):
    if "|" not in line:
        continue
    cells = [c.strip() for c in line.split("|")]
    if len(cells) < 3:
        continue
    if cells[1] == "total_timesteps":
        try:
            last_ts = float(cells[2])
            while pending:
                pairs.append((last_ts, pending.pop(0)))
        except ValueError:
            pass
    elif cells[1] == "avg_ante":
        try:
            pending.append(float(cells[2]))
        except ValueError:
            pass

if len(pairs) < 2:
    sys.exit(f"not enough eval points in {run_dir/'train.log'}")

x = np.array([p[0] for p in pairs])
y = np.array([p[1] for p in pairs])
n = len(x)
print(f"points={n}  span={x.min()/1e3:.0f}k -> {x.max()/1e3:.0f}k steps")

slope, icept = np.polyfit(x, y, 1)
r = np.corrcoef(x, y)[0, 1]
print(f"linear fit: slope={slope*1e6:+.2f} ante/Msteps  r={r:.2f}")

half = max(n // 2, 1)
for name, seg in (("first half", (x[:half], y[:half])),
                  ("second half", (x[half:], y[half:]))):
    s = np.polyfit(seg[0], seg[1], 1)[0]
    print(f"{name}: slope={s*1e6:+.2f} ante/Msteps  mean={seg[1].mean():.2f}")

ls, li = np.polyfit(np.log(x), y, 1)
proj = np.polyval([ls, li], np.log(2e7))
print(f"log-time fit: projected avg_ante @20M = {proj:.2f}")
print(f"current level: mean of last {min(5, n)} = "
      f"{y[-min(5, n):].mean():.2f}")
if n < 15:
    print(f"NOTE: {n} points is too few for a trustworthy slope — "
          f"re-run after more evals accumulate.")
