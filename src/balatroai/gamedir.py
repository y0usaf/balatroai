"""Isolated vanilla game copy.

`balatroai setup` clones the Steam Balatro install + Proton prefix into a
private game dir with ONLY the mods the bot needs (smods + balatrobot).
The user's Steam install and mod collection stay untouched, and their mods
(Talisman, Pokermon, …) can't corrupt the bot's gamestate.

Layout (default ~/.local/share/balatroai):
  Balatro/         copy of the game dir (incl. lovely version.dll)
  compatdata/      copy of the Proton prefix (profile/unlocks preserved)
    …/Balatro/Mods/{smods, balatrobot}   only these two
  balatrobot-src/  clone of coder/balatrobot (mod lua source)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

APP_ID = "2379780"
MODS_REL = Path("pfx/drive_c/users/steamuser/AppData/Roaming/Balatro/Mods")
BALATROBOT_URL = "https://github.com/coder/balatrobot"


def default_game_dir() -> Path:
    if env := os.environ.get("BALATROAI_GAME"):
        return Path(env)
    return Path.home() / ".local/share/balatroai"


def steam_root() -> Path:
    for p in (Path.home() / ".local/share/Steam", Path.home() / ".steam/steam"):
        if (p / "steamapps").is_dir():
            return p
    sys.exit("Steam not found (~/.local/share/Steam or ~/.steam/steam)")


def is_set_up(game: Path) -> bool:
    return (game / "Balatro/Balatro.exe").is_file() and (game / "compatdata" / MODS_REL / "balatrobot/balatrobot.lua").is_file()


def _detect_proton() -> Path:
    """Proton script matching the Steam prefix (read from compatdata config_info).

    balatrobot's own detection picks the alphabetically-first Proton, which is
    often NOT the one the prefix was built with — a mismatch makes Proton
    downgrade/recreate the prefix.  The prefix's config_info records the exact
    Proton Steam used.
    """
    config = steam_root() / f"steamapps/compatdata/{APP_ID}/config_info"
    if config.is_file():
        for line in config.read_text().splitlines():
            if "/files/" in line and "proton" in line.lower():
                root = line.split("/files/")[0]
                proton = Path(root) / "proton"
                if proton.is_file():
                    return proton
    # fallback: first proton in common/
    common = steam_root() / "steamapps/common"
    for d in sorted(common.iterdir()):
        p = d / "proton"
        if p.is_file() and "proton" in d.name.lower():
            return p
    sys.exit("Proton not found")


def _steam_run_wrapper(game: Path) -> Path:
    """steam-run wrapper around Proton, so the game gets an FHS env on NixOS.

    NixOS has no /lib64, so the game's ELF interpreter
    (/lib64/ld-linux-x86-64.so.2) is missing and Proton dies with
    "could not open".  steam-run provides the FHS.  balatrobot runs this
    wrapper as its Proton via BALATROBOT_LOVE_PATH.
    """
    wrapper = game / "steam-run-proton"
    proton = _detect_proton()
    if shutil.which("steam-run"):
        wrapper.write_text(f"#!/bin/sh\nexec steam-run \"{proton}\" \"$@\"\n")
    else:  # non-NixOS: Proton already has a real /lib64
        wrapper.write_text(f"#!/bin/sh\nexec \"{proton}\" \"$@\"\n")
    wrapper.chmod(0o755)
    return wrapper


def launch_env(game: Path) -> dict[str, str]:
    """Env overrides that point balatrobot serve at the isolated copy."""
    return {
        "BALATROBOT_BALATRO_PATH": str(game / "Balatro"),
        "STEAM_COMPAT_DATA_PATH": str(game / "compatdata"),
        "BALATROBOT_LOVE_PATH": str(_steam_run_wrapper(game)),
    }


def kill_prefix(game: Path) -> None:
    """wineserver -k on the isolated prefix. Proton double-forks its children
    away from the serve process group, and upstream's cleanup only targets the
    Steam prefix — so we must reap our own orphans."""
    common = steam_root() / "steamapps/common"
    for d in sorted(common.iterdir()):
        ws = d / "files/bin/wineserver"
        if "proton" in d.name.lower() and ws.is_file():
            subprocess.run(
                [str(ws), "-k"],
                env={**os.environ, "WINEPREFIX": str(game / "compatdata/pfx")},
                capture_output=True,
                timeout=15,
            )
            return


# (old, new) pairs applied to balatrobot's settings.lua in the isolated copy.
# Rendered mode should behave like vanilla-under-Steam: real dt (wall-clock
# smooth at any fps) and vsync (frames paced to the display, not a busy loop).
# Headless keeps upstream behavior: pinned fast dt, vsync off. Idempotent.
_BALATROBOT_PATCHES = [
    (
        """love.update = function(_)
    love_update(dt)
  end""",
        """love.update = function(real_dt)
    -- balatroai: rendered mode passes real dt (Steam-like wall-clock smoothness
    -- at any fps); headless keeps the pinned fast dt for determinism and speed.
    love_update(BB_SETTINGS.headless and dt or math.min(real_dt, 1.0 / 20.0))
  end""",
    ),
    (
        """  love.window.setVSync(0)
  G.SETTINGS.WINDOW = G.SETTINGS.WINDOW or {}
  G.SETTINGS.WINDOW.vsync = 0""",
        """  -- balatroai: vsync paces rendered frames to the display (Steam feel);
  -- an uncapped busy loop presents at chaotic intervals = judder.
  local vsync = BB_SETTINGS.headless and 0 or 1
  love.window.setVSync(vsync)
  G.SETTINGS.WINDOW = G.SETTINGS.WINDOW or {}
  G.SETTINGS.WINDOW.vsync = vsync""",
    ),
]


def patch_balatrobot(game: Path) -> None:
    """Apply balatroai's rendered-mode patches to the balatrobot mod copy."""
    path = game / "compatdata" / MODS_REL / "balatrobot/src/lua/settings.lua"
    if not path.is_file():
        return
    text = path.read_text()
    for old, new in _BALATROBOT_PATCHES:
        if old in text:
            text = text.replace(old, new)
    path.write_text(text)


def _window_patch_toml(w: int, h: int) -> str:
    """lovely patch pinning Balatro's 'Windowed' size to w×h.

    conf.lua opens the initial window at 0×0 (= desktop size), and
    G.FUNCS.apply_window_changes sizes a Windowed window to 80% of the
    *current* window — i.e. 80% of the monitor — ignoring screen_res
    entirely. Replace both dimension expressions with fixed values.
    """
    return f'''[manifest]
version = "1.0.0"
priority = 100

[[patches]]
[patches.pattern]
target = "functions/button_callbacks.lua"
pattern = "*QUEUED_CHANGE.screenmode == 'Windowed') and love.graphics.getWidth()*"
position = "at"
match_indent = true
payload = "(G.SETTINGS.QUEUED_CHANGE and G.SETTINGS.QUEUED_CHANGE.screenmode == 'Windowed') and {w} or G.SETTINGS.WINDOW.DISPLAYS[G.SETTINGS.WINDOW.selected_display].screen_res.w,"

[[patches]]
[patches.pattern]
target = "functions/button_callbacks.lua"
pattern = "*QUEUED_CHANGE.screenmode == 'Windowed') and love.graphics.getHeight()*"
position = "at"
match_indent = true
payload = "(G.SETTINGS.QUEUED_CHANGE and G.SETTINGS.QUEUED_CHANGE.screenmode == 'Windowed') and {h} or G.SETTINGS.WINDOW.DISPLAYS[G.SETTINGS.WINDOW.selected_display].screen_res.h,"
'''


def set_windowed(game: Path, w: int = 1280, h: int = 720) -> None:
    """Force a real w×h window in the isolated copy.

    Two parts, re-applied before every rendered launch:
      * settings.jkr: screenmode="Windowed" (fullscreen=false) — the game
        rewrites this file on exit.
      * Mods/balatroai-window/lovely.toml: pins the windowed size, which
        the game otherwise derives from the desktop resolution.
    """
    patch_dir = game / "compatdata" / MODS_REL / "balatroai-window"
    if patch_dir.parent.is_dir():
        patch_dir.mkdir(exist_ok=True)
        (patch_dir / "lovely.toml").write_text(_window_patch_toml(w, h))

    path = game / "compatdata" / MODS_REL.parent / "settings.jkr"
    if not path.is_file():
        return
    raw = path.read_bytes()
    try:
        text = zlib.decompress(raw, -15).decode()
        compressed = True
    except zlib.error:
        text = raw.decode()
        compressed = False
    text = re.sub(r'\["screenmode"\]="[^"]*"', '["screenmode"]="Windowed"', text)
    text = re.sub(
        r'\["screen_res"\]=\{[^{}]*\}',
        f'["screen_res"]={{["w"]={w},["h"]={h},}}',
        text,
    )
    out = text.encode()
    if compressed:
        co = zlib.compressobj(1, zlib.DEFLATED, -15)
        out = co.compress(out) + co.flush()
    path.write_bytes(out)


def _copytree(src: Path, dst: Path, exclude: Path | None = None) -> None:
    """rsync if available (fast re-runs), else shutil."""
    if shutil.which("rsync"):
        cmd = ["rsync", "-a", "--delete"]
        if exclude:
            cmd += ["--exclude", f"/{exclude.as_posix()}/"]
        cmd += [f"{src}/", f"{dst}/"]
        subprocess.run(cmd, check=True)
    else:
        if dst.exists():
            shutil.rmtree(dst)
        ignore = None
        if exclude:
            excl_abs = src / exclude

            def ignore(d: str, names: list[str]) -> list[str]:
                return [n for n in names if Path(d) / n == excl_abs]

        shutil.copytree(src, dst, symlinks=True, ignore=ignore)


def _find_smods(mods: Path) -> Path | None:
    for name in ("smods", "Steamodded", "steamodded-main"):
        if (mods / name).is_dir():
            return mods / name
    return None


def setup(game: Path, balatrobot_src: Path | None = None) -> None:
    steam = steam_root()
    src_game = steam / "steamapps/common/Balatro"
    src_compat = steam / f"steamapps/compatdata/{APP_ID}"
    if not (src_game / "Balatro.exe").is_file():
        sys.exit(f"Balatro not found at {src_game}")
    if not (src_game / "version.dll").is_file():
        sys.exit(f"lovely version.dll missing in {src_game} — install lovely-injector first")
    if not src_compat.is_dir():
        sys.exit(f"Proton prefix not found at {src_compat} — run Balatro once via Steam")

    game.mkdir(parents=True, exist_ok=True)

    print(f"copying game     {src_game} -> {game / 'Balatro'}")
    _copytree(src_game, game / "Balatro")

    print(f"copying prefix   {src_compat} -> {game / 'compatdata'}  (without Mods)")
    _copytree(src_compat, game / "compatdata", exclude=MODS_REL)

    mods_dst = game / "compatdata" / MODS_REL
    if mods_dst.exists():
        shutil.rmtree(mods_dst)
    mods_dst.mkdir(parents=True)

    smods = _find_smods(src_compat / MODS_REL)
    if smods is None:
        sys.exit(f"smods/Steamodded not found in {src_compat / MODS_REL}")
    print(f"installing mod   {smods.name}")
    shutil.copytree(smods, mods_dst / smods.name, symlinks=True)

    if balatrobot_src is None:
        balatrobot_src = game / "balatrobot-src"
        if not balatrobot_src.is_dir():
            print(f"cloning          {BALATROBOT_URL}")
            subprocess.run(
                ["git", "clone", "--depth", "1", BALATROBOT_URL, str(balatrobot_src)],
                check=True,
            )
    print("installing mod   balatrobot")
    bb = mods_dst / "balatrobot"
    bb.mkdir()
    shutil.copy(balatrobot_src / "balatrobot.json", bb)
    shutil.copy(balatrobot_src / "balatrobot.lua", bb)
    shutil.copytree(balatrobot_src / "src", bb / "src", symlinks=True)

    print("installing mod   balatroai-window (1280×720 windowed)")
    set_windowed(game)

    print("patching mod     balatrobot (real dt + vsync in rendered mode)")
    patch_balatrobot(game)

    print(f"\nready: {game}\nmods: {', '.join(sorted(p.name for p in mods_dst.iterdir()))}")
