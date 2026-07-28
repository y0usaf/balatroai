"""In-process simulator client backed by jackdaw.

jackdaw's ``SimBackend.handle(method, params)`` implements the same
balatrobot JSON-RPC surface the live game speaks (same methods, same
gamestate serialization — validated scenario-by-scenario against a live
instance).  This adapter gives it the ``Client`` interface so ``Runner``
and every bot work unchanged: train/bench on the sim at thousands of
games per second, demo the same bot on the real game.

jackdaw is an optional dependency (``sim`` extra): the stdlib-only core
stays intact, and this module import-fails with instructions rather than
poisoning ``balatroai`` imports.
"""

from __future__ import annotations

from .client import RPCError


class SimClient:
    """Drop-in for :class:`balatroai.client.Client`, no server needed.

    Each ``SimClient`` owns one game state, like one balatrobot instance.
    """

    def __init__(self) -> None:
        try:
            from jackdaw.bridge.backend import RPCError as _JackdawRPCError
            from jackdaw.bridge.backend import SimBackend
        except ImportError as e:  # pragma: no cover - environment dependent
            raise SystemExit(
                "the sim backend needs jackdaw (not installed).\n"
                "  uv sync --extra sim      # dev checkout\n"
                "or run against the live game instead (default backend)."
            ) from e
        self._backend = SimBackend()
        self._rpc_error = _JackdawRPCError

    def call(self, method: str, params: dict | None = None) -> dict:
        try:
            return self._backend.handle(method, params or {})
        except self._rpc_error as e:
            raise RPCError("SIM", str(e)) from e

    # -- parity with Client's readiness API (the sim is always ready) --

    def ready(self) -> bool:
        return True

    def wait_ready(self, timeout: float = 0.0, interval: float = 0.0) -> bool:
        return True
