"""Minimal JSON-RPC 2.0 client for the balatrobot server. Stdlib only."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class RPCError(Exception):
    """Error returned by the balatrobot server."""

    def __init__(self, name: str, message: str):
        self.name = name
        self.message = message
        super().__init__(f"{name}: {message}")


class ConnectionDown(Exception):
    """Server unreachable."""


class Client:
    def __init__(self, host: str = "127.0.0.1", port: int = 12346, timeout: float = 120.0):
        self.url = f"http://{host}:{port}"
        self.timeout = timeout
        self._id = 0

    def call(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        payload = json.dumps(
            {"jsonrpc": "2.0", "method": method, "params": params or {}, "id": self._id}
        ).encode()
        req = urllib.request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, OSError) as e:
            raise ConnectionDown(str(e)) from e
        if "error" in data:
            err = data["error"]
            name = (err.get("data") or {}).get("name", "ERROR")
            raise RPCError(name, err.get("message", "unknown error"))
        return data["result"]

    def ready(self) -> bool:
        try:
            return self.call("health").get("status") == "ok"
        except (ConnectionDown, RPCError):
            return False

    def wait_ready(self, timeout: float = 180.0, interval: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.ready():
                return True
            time.sleep(interval)
        return False
