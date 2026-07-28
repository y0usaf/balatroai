"""Print a snapshot of a running (or finished) training job.

Reads only the run directory, so it never touches the trainer process:

    nix develop -c .venv/bin/python scripts/train_status.py runs/pointer_v4

Shows current step, throughput, the ante/win trend over the last window, and
which checkpoints exist. Pass --watch to refresh every 30 s.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

STEP_RE = re.compile(
    r"step\s+([\d,]+)\s+sps\s+([\d,]+)\s+c/u\s+([\d.]+)/([\d.]+)s"
    r"(?:\s+ante\s+([\d.]+)\s+\(max\s+(\d+)\)\s+win\s+([\d.]+)%)?"
)


def parse(log: Path) -> list[dict]:
    rows = []
    for line in log.read_text(errors="ignore").splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        rows.append({
            "step": int(m.group(1).replace(",", "")),
            "sps": int(m.group(2).replace(",", "")),
            "collect": float(m.group(3)),
            "update": float(m.group(4)),
            "ante": float(m.group(5)) if m.group(5) else None,
            "max_ante": int(m.group(6)) if m.group(6) else None,
            "win": float(m.group(7)) if m.group(7) else None,
        })
    return rows


def mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def report(run_dir: Path, log_name: str, window: int) -> None:
    log = run_dir / log_name
    if not log.exists():
        print(f"no log at {log}")
        return
    rows = parse(log)
    if not rows:
        print(f"{log}: no step lines yet (still starting up?)")
        return

    last = rows[-1]
    recent = rows[-window:]
    prev = rows[-2 * window : -window] or recent
    ante_now, ante_prev = mean([r["ante"] for r in recent]), mean([r["ante"] for r in prev])
    sps = mean([float(r["sps"]) for r in recent]) or 0.0
    age = time.time() - log.stat().st_mtime

    print(f"run        {run_dir}")
    print(f"step       {last['step']:,}   ({sps:,.0f} steps/s over last {len(recent)} iters)")
    print(f"collect/upd{last['collect']:>6.1f}s /{last['update']:>5.1f}s")
    if ante_now is not None:
        trend = "" if ante_prev is None else f"  (was {ante_prev:.2f})"
        print(f"mean ante  {ante_now:.2f}{trend}   best seen "
              f"{max(r['max_ante'] or 0 for r in rows)}")
        print(f"win rate   {mean([r['win'] for r in recent]) or 0:.1f}%")
    cks = sorted(run_dir.glob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime)
    print(f"checkpoints{len(cks):>4}   latest: {cks[-1].name if cks else '-'}")
    print(f"log age    {age:,.0f}s  {'(STALLED?)' if age > 300 else ''}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Training run status")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--log-name", default="train.log")
    ap.add_argument("--window", type=int, default=20, help="iterations to average")
    ap.add_argument("--watch", action="store_true", help="refresh every 30 s")
    args = ap.parse_args()

    while True:
        report(args.run_dir, args.log_name, args.window)
        if not args.watch:
            return
        print("-" * 60, flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
