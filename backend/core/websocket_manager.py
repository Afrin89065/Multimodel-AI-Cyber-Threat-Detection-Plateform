from fastapi import WebSocket
from typing import Dict, List
from loguru import logger
import json

class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, List[WebSocket]] = {}

    async def connect(self, ws: WebSocket, room: str = "soc"):
        await ws.accept()
        self.active.setdefault(room, []).append(ws)
        logger.info(f"WebSocket connected to room '{room}' — total: {len(self.active[room])}")

    def disconnect(self, ws: WebSocket, room: str = "soc"):
        if room in self.active:
            try:
                self.active[room].remove(ws)
            except ValueError:
                pass

    async def broadcast(self, data: dict, room: str = "soc"):
        if room not in self.active:
            return
        msg = json.dumps(data, default=str)
        dead = []
        for ws in self.active[room]:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            try:
                self.active[room].remove(ws)
            except ValueError:
                pass

ws_manager = ConnectionManager()