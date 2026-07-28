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
| 06 bare core must boot | follows | random bot + runner completes a game with zero heuristics; `nix flake check` runs the offline bare-core check (poker selftest, CLI, registry); the jackdaw `sim` extra is opt-in and lazily imported, so the bare core boots without it |
| 07 nix source of truth | follows (with gap) | flake builds the package and runs checks; `uv run` is the day-to-day dev loop; the `sim` extra resolves via uv git-rev pin to our fork (`github.com:y0usaf/jackdaw-balatro`), not the flake — fold into the flake if the sim backend graduates from optional |

## Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Runtime deps | stdlib-only **core**; jackdaw as opt-in `sim` extra | keeps the flake trivial and the core auditable; the sim backend is lazily imported so `balatroai` runs without it (divergence from the original "stdlib only" absolute — recorded here) |
| balatrobot | external process, not vendored | upstream moves fast; we speak only its versioned JSON-RPC surface |
| Fast simulator | adopt [jackdaw](https://github.com/TylerFlar/jackdaw-balatro), not build | its `SimBackend.handle(method, params)` speaks the same balatrobot RPC surface (methods + gamestate serialization), so Runner/bots run unchanged on either backend; ~1500 games/sec |
| jackdaw source | own fork `github.com:y0usaf/jackdaw-balatro` (mirrored on forgejo at y0usaf-server:3000), rev-pinned in `pyproject.toml` | carries our 11 engine fixes from validating it against live Balatro 1.0.1o + smods (253/254 scenarios pass); upstream PR pending — switch back to upstream once merged |
| Fidelity target | vanilla **+ Steamodded**, not vanilla | balatrobot requires smods, and smods silently rewrites game logic (spectral RNG order, etc.) — "1:1 with vanilla" and "1:1 with what the bots actually play" are different targets; we validate against the latter |
| First bot | heuristic, not ML | immediately watchable; becomes the RL baseline and the BC teacher |
| Score model | estimate only (base chips×mult + card chips) | bots stay sim-agnostic; exact scoring lives in jackdaw, not in bot heuristics |
| Training against live game | rejected for RL | Proton instances are 2–4 orders of magnitude too slow; RL runs on jackdaw |

## Architecture

```
src/balatroai/
  client.py    core — JSON-RPC client (urllib), health/wait_ready
  sim.py       core — SimClient: same interface, backed by in-process jackdaw
  runner.py    core — game loop, per-state fallbacks, step watchdog
  poker.py     core — hand classification + best-subset scoring (offline-testable)
  bots/
    __init__.py   core — Action type, Bot protocol, registry (the extension surface)
    random_bot.py policy — bare-core baseline
    heuristic.py  policy — best-hand play, discard fishing, joker/planet shopping
  cli.py       policy — watch / run subcommands, optional `--launch` of balatrobot
```

Both clients expose `call(method, params) -> gamestate dict` over the same
balatrobot RPC vocabulary; `Runner` and bots are backend-blind.  `run`
defaults to the sim (train/bench fast), `watch` is always the live game
(demo what was trained).

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
- [x] P2 — fast simulator. *Adopted jackdaw instead of building: validated
  scenario-by-scenario against live Balatro 1.0.1o + smods on this machine
  (253/254 pass; 11 engine fixes on the `live-parity` branch), wired in as
  `run --backend sim`. Exceeds the original accept criterion (≥95% score
  match) — remaining follow-up: replay-diff logged full games as a
  regression harness when versions bump.*
- [ ] P3 — RL trainer daemon. *Accept: `balatroai train --workers N` runs
  unattended; `balatroai watch --checkpoint best` plays the live game;
  policy beats heuristic's win rate on the P1 bench.*
- [ ] P4 — stake climbing. *Accept: bench matrix across stakes; red stake
  win rate > 0.*
