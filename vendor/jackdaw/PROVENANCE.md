# Provenance

Vendored copy of the `jackdaw` Balatro simulator, so balatroAI builds without
access to a private remote.

- Upstream: <https://github.com/TylerFlar/jackdaw-balatro> — MIT,
  Copyright (c) 2026 Tyler Flar. `LICENSE` is preserved verbatim.
- Fork vendored here: <https://github.com/y0usaf/jackdaw-balatro>, branch
  `live-parity` (the fork carrying the engine fixes validated 1:1 against live
  Balatro 1.0.1o + smods).
- Revision: `b3730688f342039fc767776e536cdb294d6c5ac4`, exported with
  `git archive` on 2026-09-19, so every file here is byte-identical to that
  commit. Uncommitted local edits present in the maintainer's clone are
  deliberately excluded.
- Pruned from the export: `.github/`, `.gitignore`, `.python-version`,
  `tests/`, `scripts/`, `build/`, `*.egg-info/`, `uv.lock`, `shard*.txt`,
  `probe_*.py`, `shard_runner.py`. Kept: `jackdaw/` (package, incl.
  `engine/data/*.json`), `LICENSE`, `README.md`, `docs/`, `pyproject.toml`.
- Updating: re-export a newer revision of the fork over this directory and
  update the revision and date above; balatroAI's `[tool.uv.sources]` points at
  `path = "vendor/jackdaw"`, so no lockfile URL changes are needed.
