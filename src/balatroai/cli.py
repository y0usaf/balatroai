"""balatroai CLI — watch a bot play, or run batches for stats.

  balatroai watch [--bot heuristic] [--seed S] [--launch]
  balatroai run --games 10 [--bot heuristic] [--launch]
"""

from __future__ import annotations

import argparse
import atexit
import random
import shutil
import string
import subprocess
import sys

from .bots import Action, get_bot
from .client import Client
from .runner import Runner

SUITS = {"H": "♥", "D": "♦", "C": "♣", "S": "♠"}


def fmt_card(c: dict) -> str:
    v = c.get("value", {})
    rank, suit = v.get("rank"), v.get("suit")
    if rank and suit:
        return f"{rank}{SUITS.get(suit, suit)}"
    return c.get("label", "?")


def random_seed() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


def launch_server(port: int, headless: bool) -> subprocess.Popen:
    uvx = shutil.which("uvx")
    if not uvx:
        sys.exit("--launch needs `uvx` (uv) on PATH; or start `uvx balatrobot serve` yourself")
    cmd = [uvx, "balatrobot", "serve", "--port", str(port)]
    cmd += ["--headless", "--fast"] if headless else ["--gamespeed", "2"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(proc.terminate)
    return proc


def connect(args, headless: bool) -> Client:
    client = Client(args.host, args.port)
    if args.launch:
        print(f"launching balatrobot on port {args.port}…")
        launch_server(args.port, headless)
    if not client.wait_ready(timeout=180 if args.launch else 5):
        sys.exit(
            f"no balatrobot server on {args.host}:{args.port}\n"
            "start one with:  uvx balatrobot serve   (or pass --launch)"
        )
    return client


def watch_emit(state: dict, action: Action, error: str | None) -> None:
    ante = state.get("ante_num", "?")
    rnd_num = state.get("round_num", "?")
    money = state.get("money", 0)
    rnd = state.get("round", {})
    prefix = f"[a{ante} r{rnd_num} ${money}]"

    line = f"{prefix} {action.note or action.method}"
    if action.method in ("play", "discard"):
        chips = rnd.get("chips", 0)
        line += (
            f"  → {chips} chips"
            f" · {rnd.get('hands_left', '?')}h/{rnd.get('discards_left', '?')}d left"
        )
    if state.get("state") == "SELECTING_HAND":
        cards = state.get("hand", {}).get("cards", [])
        line += "\n" + " " * 4 + "hand: " + " ".join(fmt_card(c) for c in cards)
    if error:
        line += f"\n    !! {error} (fell back)"
    print(line, flush=True)


def cmd_watch(args) -> int:
    client = connect(args, headless=False)
    bot = get_bot(args.bot)
    seed = args.seed or random_seed()
    print(f"bot={bot.name} deck={args.deck} stake={args.stake} seed={seed}\n")
    result = Runner(client, bot, emit=watch_emit).play_game(args.deck, args.stake, seed)
    verdict = "WON 🎉" if result.won else "lost"
    print(f"\n{verdict} — ante {result.ante}, round {result.round}, "
          f"seed {result.seed}, {result.steps} actions")
    return 0 if result.won else 1


def cmd_run(args) -> int:
    client = connect(args, headless=True)
    bot = get_bot(args.bot)
    seeds = [args.seed] if args.seed else [random_seed() for _ in range(args.games)]
    if args.seed and args.games > 1:
        seeds = [args.seed] * args.games

    runner = Runner(client, bot)
    results = []
    for i, seed in enumerate(seeds, 1):
        r = runner.play_game(args.deck, args.stake, seed)
        results.append(r)
        mark = "W" if r.won else "L"
        print(f"game {i:>3}/{len(seeds)}  {mark}  ante {r.ante}  round {r.round}  seed {r.seed}",
              flush=True)

    wins = sum(r.won for r in results)
    antes = sorted(r.ante for r in results)
    print(f"\n{bot.name}: {wins}/{len(results)} wins "
          f"({100 * wins / len(results):.0f}%) · "
          f"ante avg {sum(antes) / len(antes):.1f} · "
          f"median {antes[len(antes) // 2]} · best {antes[-1]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="balatroai", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--bot", default="heuristic", help="bot name (default: heuristic)")
        p.add_argument("--deck", default="RED")
        p.add_argument("--stake", default="WHITE")
        p.add_argument("--seed", default=None, help="run seed (default: random)")
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=12346)
        p.add_argument("--launch", action="store_true",
                       help="spawn `uvx balatrobot serve` and wait for it")

    p_watch = sub.add_parser("watch", help="watch a bot play one rendered game")
    common(p_watch)
    p_watch.set_defaults(func=cmd_watch)

    p_run = sub.add_parser("run", help="run N games and print stats")
    common(p_run)
    p_run.add_argument("--games", type=int, default=10)
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
