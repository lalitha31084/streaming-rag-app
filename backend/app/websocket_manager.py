"""
WebSocket connection manager.

Maintains a set of active WebSocket connections and provides helpers
for broadcasting messages. Each connection is fully independent
(no crosstalk between concurrent user sessions).
"""

import logging
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Track active WebSocket connections for observability."""

    def __init__(self):
        self._active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.append(ws)
        logger.info("Client connected. Active sessions: %d", len(self._active))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._active:
            self._active.remove(ws)
        logger.info("Client disconnected. Active sessions: %d", len(self._active))

    @property
    def active_count(self) -> int:
        return len(self._active)

    async def broadcast_json(self, data: dict) -> None:
        """Send a JSON message to every connected client."""
        for ws in self._active:
            try:
                await ws.send_json(data)
            except Exception:
                pass
