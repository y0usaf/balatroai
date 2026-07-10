# balatroAI

Bots that play Balatro through the [balatrobot](https://github.com/coder/balatrobot)
JSON-RPC API. `watch` a rendered game with live narration, or `run` headless
batches for stats. See [DESIGN.md](./DESIGN.md) for the roadmap (fast sim → RL).

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
# terminal 1 — the game (rendered, 2x speed)
uvx balatrobot serve --gamespeed 2

# terminal 2 — watch the heuristic bot play
nix run . -- watch                 # or: uv run balatroai watch
```

Or let balatroai spawn the server itself:

```bash
nix run . -- watch --launch
```

## Commands

```bash
balatroai watch [--bot heuristic|random] [--seed ABCD1234] [--deck RED] [--stake WHITE]
balatroai run --games 20 [--bot heuristic] [--launch]   # headless stats
```

## Development

```bash
nix develop          # python + uv + ruff
nix flake check      # bare-core check (offline: poker selftest, CLI, registry)
uv run balatroai --help
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
