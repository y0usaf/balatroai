# balatroAI — design & roadmap

Bots that play Balatro through the [balatrobot](https://github.com/coder/balatrobot)
JSON-RPC API. Exists to answer "how good can an autonomous Balatro player get,
and can I watch it?" — watchable first, trainable later.

## Vision

- **As watchable as a Twitch bot run** — `balatroai watch` narrates every
  decision of a rendered game in real time.
- **As benchable as a test suite** — `balatroai run --games N` gives win
  rate / ante distribution on fixed seeds, so bots are comparable.
- **Eventually as trainable as an RL env** — parallel headless instances
  feeding a trainer daemon; heuristic bot is the baseline it must beat.

Reference: balatrobot's OpenRPC spec (`src/lua/utils/openrpc.json` upstream)
is the source of truth for the wire schema.

## Doctrine conformance

| Doctrine | Status | Notes |
|---|---|---|
| 01 extension-first core | follows (lite) | the unit is the *bot*; core (client/runner/cli) contains no bot-specific logic; built-in bots use the same `register` API a user bot would |
| 02 snapshot in, actions out | follows | bots see gamestate dicts, return `Action`; runner owns all I/O; watchdog = `max_steps=3000` + per-state fallbacks that always advance the game |
| 03 daemon + thin client | diverges (for now) | balatrobot's server is the daemon; balatroai is a single-process client. A trainer daemon arrives with the RL phase (P3) |
| 04 declarative front, idempotent executor | n/a | no system state is managed |
| 05 one declaration mechanism | follows | every bot declared via `bots.register`; no hand-wired dispatch |
| 06 bare core must boot | follows | random bot + runner completes a game with zero heuristics; `nix flake check` runs the offline bare-core check (poker selftest, CLI, registry) |
| 07 nix source of truth | follows | flake builds the package and runs checks; `uv run` is the dev-loop fallback (game server can't run in the sandbox) |

## Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Runtime deps | stdlib only | keeps the flake trivial and the core auditable; JSON-RPC over HTTP needs nothing more |
| balatrobot | external process, not vendored | upstream moves fast; we speak only its versioned JSON-RPC surface |
| First bot | heuristic, not ML | immediately watchable; becomes the RL baseline and the BC teacher |
| Score model | estimate only (base chips×mult + card chips) | exact simulation of joker interactions is the P2 fast-sim project, not a bot concern |
| Training against live game | rejected for RL | Proton instances are 2–4 orders of magnitude too slow; RL waits for the fast sim (P2) |

## Architecture

```
src/balatroai/
  client.py    core — JSON-RPC client (urllib), health/wait_ready
  runner.py    core — game loop, per-state fallbacks, step watchdog
  poker.py     core — hand classification + best-subset scoring (offline-testable)
  bots/
    __init__.py   core — Action type, Bot protocol, registry (the extension surface)
    random_bot.py policy — bare-core baseline
    heuristic.py  policy — best-hand play, discard fishing, joker/planet shopping
  cli.py       policy — watch / run subcommands, optional `--launch` of balatrobot
```

## Extension surface contract

- **Read path**: `Bot.act(state)` receives the full balatrobot gamestate dict.
  Treat it as immutable; it is replaced wholesale each step.
- **Write path**: return one `Action(method, params, note)`. The runner
  executes it; a rejected action triggers the state's fallback, never a retry.
- **Watchdog**: 3000 actions per game, 120 s per RPC.
- **Bot state**: instance attributes only; a bot instance lives for one
  `Runner`, which may span many games.

## Deferred (and why)

- **Trainer daemon + parallel workers** — needs the fast sim to be worth it;
  orchestration alone is plumbing (balatrobot's `manager.py` already does it).
- **Fast native simulator** (P2) — the real enabler for RL; weeks of joker
  fidelity work, so it waits until the heuristic ceiling is measured.
- **Tarot/spectral usage, voucher buying, blind skipping, rearrange** —
  each is a heuristic-bot upgrade, not core work; add when bench numbers exist.
- **TUI/leaderboard** — plain stdout narration is enough to be fun; revisit
  after P1.

## Roadmap

- [x] P0 — watchable heuristic bot. *Accept: `balatroai watch` narrates a full
  rendered game to GAME_OVER against a live balatrobot server.*
- [ ] P1 — benchmarking. *Accept: `balatroai run --games 50` on fixed seed set
  prints win rate + ante distribution; heuristic beats random with p < 0.05.*
- [ ] P2 — fast simulator. *Accept: sim replays a logged balatrobot game and
  matches the real score for ≥95% of hands on white stake.*
- [ ] P3 — RL trainer daemon. *Accept: `balatroai train --workers N` runs
  unattended; `balatroai watch --checkpoint best` plays the live game;
  policy beats heuristic's win rate on the P1 bench.*
- [ ] P4 — stake climbing. *Accept: bench matrix across stakes; red stake
  win rate > 0.*
