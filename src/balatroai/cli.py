"""balatroai CLI — watch a bot play, or run batches for stats.

  balatroai watch [--bot heuristic] [--seed S] [--launch]
  balatroai run --games 10 [--bot heuristic] [--launch]
"""

from __future__ import annotations

import argparse
import atexit
import os
import random
import shutil
import string
import subprocess
import sys
from pathlib import Path

from . import gamedir
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


# Fixed seed set for the P1 bench — reproducible, both bots share it.
BENCH_SEEDS = [f"B{i:07d}" for i in range(50)]


def launch_server(port: int, headless: bool, game: Path | None, gamespeed: int = 2) -> subprocess.Popen:
    uvx = shutil.which("uvx")
    if not uvx:
        sys.exit("--launch needs `uvx` (uv) on PATH; or start `uvx balatrobot serve` yourself")
    cmd = [uvx, "balatrobot", "serve", "--port", str(port)]
    # Upstream balatrobot pins love.update dt (rendered 1/60, headless 4.99/60),
    # making game speed = achieved_fps/60 — fps wobble becomes time-warp judder.
    # gamedir.patch_balatrobot() gives rendered mode vanilla Steam behavior: real
    # dt (smooth at any fps) + vsync (frames paced to the display; an uncapped
    # busy loop presents at chaotic intervals = judder even with correct dt).
    # Headless stays pinned-fast/vsync-off. Vanilla main.lua clamps a nil
    # G.FPS_CAP to 500 — and --fast sets exactly nil — so we skip --fast and
    # pass an explicit huge cap (vsync does the pacing in rendered mode).
    # gamespeed is G.SETTINGS.GAMESPEED directly (divides animation durations).
    # --no-reduced-motion restores card wobble/bounce (upstream disables it).
    if headless:
        cmd += ["--headless", "--gamespeed", "10", "--animation-fps", "60", "--fps-cap", "999999"]
    else:
        cmd += ["--gamespeed", str(gamespeed), "--animation-fps", "60", "--fps-cap", "999999",
                "--no-reduced-motion"]
    env = os.environ.copy()
    if not headless:
        # session forces vblank sync off globally; the game wants it on
        env["__GL_SYNC_TO_VBLANK"] = "1"
    if game is not None:
        env.update(gamedir.launch_env(game))
        if not headless:
            gamedir.set_windowed(game)
        gamedir.patch_balatrobot(game)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)

    def stop() -> None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        if game is not None:
            gamedir.kill_prefix(game)  # reap Proton orphans in our prefix

    atexit.register(stop)
    return proc


def connect(args, headless: bool) -> Client:
    client = Client(args.host, args.port)
    if args.launch:
        game = Path(args.game_dir) if args.game_dir else gamedir.default_game_dir()
        if gamedir.is_set_up(game):
            print(f"launching balatrobot on port {args.port} (isolated copy: {game})…")
        else:
            game = None
            print(f"launching balatrobot on port {args.port} (Steam install — run "
                  f"`balatroai setup` for an isolated vanilla copy)…")
        launch_server(args.port, headless, game, args.gamespeed)
    if not client.wait_ready(timeout=180 if args.launch else 5):
        sys.exit(
            f"no balatrobot server on {args.host}:{args.port}\n"
            "start one with:  uvx balatrobot serve   (or pass --launch)"
        )
    return client


def cmd_setup(args) -> int:
    game = Path(args.game_dir) if args.game_dir else gamedir.default_game_dir()
    src = Path(args.balatrobot_src) if args.balatrobot_src else None
    gamedir.setup(game, src)
    return 0


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
    if args.backend == "sim":
        from .sim import SimClient
        client = SimClient()
        print("backend: jackdaw sim (in-process, no live game)")
    else:
        client = connect(args, headless=True)

    seeds = [args.seed] * args.games if args.seed else BENCH_SEEDS[: args.games]

    bots = [get_bot(args.bot)]
    if args.compare:
        bots.append(get_bot("random"))

    for bot in bots:
        runner = Runner(client, bot)
        rs = [runner.play_game(args.deck, args.stake, s) for s in seeds]
        avg = sum(r.ante for r in rs) / len(rs)
        print(f"{bot.name}: avg ante {avg:.2f} ({len(rs)} games)")
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
        p.add_argument("--gamespeed", type=int, default=2,
                       help="rendered game speed multiplier (default: 2; headless always runs fast)")
        p.add_argument("--launch", action="store_true",
                       help="spawn `uvx balatrobot serve` and wait for it")
        p.add_argument("--game-dir", default=None,
                       help="isolated game copy (default: ~/.local/share/balatroai)")

    p_setup = sub.add_parser(
        "setup", help="copy Steam Balatro + prefix into an isolated vanilla install")
    p_setup.add_argument("--game-dir", default=None)
    p_setup.add_argument("--balatrobot-src", default=None,
                         help="path to a balatrobot checkout (default: clone from GitHub)")
    p_setup.set_defaults(func=cmd_setup)

    p_watch = sub.add_parser("watch", help="watch a bot play one rendered game")
    common(p_watch)
    p_watch.set_defaults(func=cmd_watch)

    p_run = sub.add_parser("run", help="run N games and print stats")
    common(p_run)
    p_run.add_argument("--games", type=int, default=50)
    p_run.add_argument("--compare", action="store_true",
                       help="also run the random bot on the same seeds and print a sign-test p-value")
    p_run.add_argument("--backend", choices=("sim", "live"), default="sim",
                       help="sim = in-process jackdaw simulator (fast, default); "
                            "live = real game via balatrobot")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
