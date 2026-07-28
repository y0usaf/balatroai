# balatroAI

Bots that play Balatro. `watch` a rendered game with live narration (via the
[balatrobot](https://github.com/coder/balatrobot) JSON-RPC API), or `run`
batches for stats on an in-process simulator
([jackdaw](https://github.com/TylerFlar/jackdaw-balatro), ~1500 games/sec,
validated 1:1 against the live game — see DESIGN.md).  Both backends speak
the same RPC surface, so every bot runs unchanged on either: **train on the
sim, demo on the real game.**  See [DESIGN.md](./DESIGN.md) for the roadmap.

## Prerequisites (one-time)

Balatro (Steam) plus the balatrobot mod stack in the Proton prefix:

1. [Lovely Injector](https://github.com/ethangreen-dev/lovely-injector) —
   `version.dll` (Windows build) into
   `~/.local/share/Steam/steamapps/common/Balatro/`
2. [Steamodded](https://github.com/Steamodded/smods) and
   [balatrobot](https://github.com/coder/balatrobot) into
   `~/.local/share/Steam/steamapps/compatdata/2379780/pfx/drive_c/users/steamuser/AppData/Roaming/Balatro/Mods/`

Full instructions: https://coder.github.io/balatrobot/

## Quickstart

```bash
# stats on the simulator — no game needed, thousands of games/sec
uv sync --extra sim
uv run balatroai run --games 100 --bot heuristic

# watch the same bot on the real game (rendered, 2x speed)
uvx balatrobot serve --gamespeed 2    # terminal 1
nix run . -- watch                    # terminal 2; or: uv run balatroai watch
```

Or let balatroai spawn the server itself:

```bash
nix run . -- watch --launch
```

## Commands

```bash
balatroai watch [--bot heuristic|random] [--seed ABCD1234] [--deck RED] [--stake WHITE] [--gamespeed 8]
balatroai run --games 20 [--bot heuristic]              # sim backend (default, fast)
balatroai run --games 20 --backend live [--launch]      # real game via balatrobot
```

The sim backend needs the `sim` extra (jackdaw, fetched from our fork
`github.com:y0usaf/jackdaw-balatro`, `live-parity` branch, rev-pinned —
carries our fixes from validating it against live Balatro + smods).  The
core stays stdlib-only; without the extra, `--backend live` works as before.

## Development

```bash
nix develop          # python + uv + ruff + GPU lib paths
nix flake check      # bare-core check (offline: poker selftest, CLI, registry)
uv run balatroai --help
uv sync --extra train   # torch + sb3-contrib for RL
```

GPU training (CUDA, torch wheels) needs `LD_LIBRARY_PATH` covering the
nix-ld set + `/run/opengl-driver/lib` (NVIDIA userspace libs) — `nix develop`
exports this for you; outside it, run:

```bash
export LD_LIBRARY_PATH="$NIX_LD_LIBRARY_PATH:/run/opengl-driver/lib"
```

## Writing a bot

```python
from balatroai.bots import Action, register

@register
class MyBot:
    name = "mybot"
    def act(self, state: dict) -> Action:
        ...
        return Action("play", {"cards": [0, 1]}, note="yolo")
```

Then `balatroai watch --bot mybot`. Contract: snapshot in, one action out —
see DESIGN.md § extension surface.
