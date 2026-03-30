"""FastAPI server — single port for HTTP frame upload and WebSocket control plane.

Runs in a background thread with its own asyncio event loop. The PyQt6 app
communicates with it through thread-safe callbacks and asyncio.run_coroutine_threadsafe.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import uvicorn
from fastapi import FastAPI, Form, UploadFile, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class FrameData:
    """A frame received from the extension."""
    image_bytes: bytes
    timestamp: str
    tab_url: str
    received_at: float = field(default_factory=time.time)


@dataclass
class ExtensionState:
    """Last known state of the connected extension."""
    connected: bool = False
    mode: str = "idle"
    muted: bool = False
    interval_sec: float = 2.0
    last_heartbeat: float = 0.0


# ── Server ────────────────────────────────────────────────────────────────────

class HushServer:
    """FastAPI server managing extension communication.

    Callbacks:
        on_frame: called (on the server's event loop thread) when a frame arrives.
        on_override: called when the extension reports a user mute/unmute override.
        on_connection_change: called when the extension connects or disconnects.
    """

    def __init__(
        self,
        port: int = 9099,
        on_frame: Optional[Callable[[FrameData], None]] = None,
        on_override: Optional[Callable[[bool], None]] = None,
        on_connection_change: Optional[Callable[[bool], None]] = None,
    ):
        self.port = port
        self.on_frame = on_frame
        self.on_override = on_override
        self.on_connection_change = on_connection_change

        self.extension = ExtensionState()
        self._ws: Optional[WebSocket] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[uvicorn.Server] = None
        self._app = self._create_app()

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="Hush", docs_url=None, redoc_url=None)

        @app.post("/frame")
        async def receive_frame(
            frame: UploadFile,
            timestamp: str = Form(""),
            tab_url: str = Form(""),
        ):
            image_bytes = await frame.read()
            if not image_bytes:
                return {"ok": False, "error": "empty frame"}

            logger.info(
                "Frame received: %d bytes, timestamp=%s", len(image_bytes), timestamp
            )
            frame_data = FrameData(
                image_bytes=image_bytes,
                timestamp=timestamp,
                tab_url=tab_url,
            )
            if self.on_frame:
                try:
                    self.on_frame(frame_data)
                except Exception:
                    logger.exception("Error in frame callback")
                    return {"ok": False, "error": "internal error"}
            return {"ok": True}

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self._ws = websocket
            self.extension.connected = True
            self.extension.last_heartbeat = time.time()
            logger.info("Extension connected")
            if self.on_connection_change:
                self.on_connection_change(True)

            try:
                while True:
                    data = await websocket.receive_json()
                    self._handle_extension_message(data)
            except WebSocketDisconnect:
                pass
            except Exception as e:
                logger.warning("WebSocket error: %s", e)
            finally:
                self._ws = None
                self.extension.connected = False
                logger.info("Extension disconnected")
                if self.on_connection_change:
                    self.on_connection_change(False)

        return app

    def _handle_extension_message(self, data: dict):
        event = data.get("event")
        logger.debug("Extension message: %s", data)
        if event == "status":
            self.extension.mode = data.get("mode", self.extension.mode)
            self.extension.muted = data.get("muted", self.extension.muted)
            self.extension.interval_sec = data.get(
                "interval_sec", self.extension.interval_sec
            )
            self.extension.last_heartbeat = time.time()
        elif event == "override":
            muted = data.get("muted", False)
            self.extension.muted = muted
            if self.on_override:
                self.on_override(muted)
        else:
            logger.warning("Unknown extension event: %s", data)

    # ── Commands to extension ─────────────────────────────────────────────────

    def send_command(self, cmd: dict):
        """Send a JSON command to the extension. Thread-safe."""
        if self._loop and self._ws:
            asyncio.run_coroutine_threadsafe(
                self._send_json(cmd), self._loop
            )

    async def _send_json(self, cmd: dict):
        if self._ws:
            try:
                await self._ws.send_json(cmd)
            except Exception as e:
                logger.warning("Failed to send command: %s", e)

    def set_mode(self, mode: str, interval_sec: float = 2.0):
        """Tell the extension to start or stop capturing."""
        cmd = {"cmd": "set_mode", "mode": mode}
        if mode == "active":
            cmd["interval_sec"] = interval_sec
        logger.info("Sending command: %s", cmd)
        self.send_command(cmd)

    def mute(self):
        logger.info("Sending command: mute")
        self.send_command({"cmd": "mute"})

    def unmute(self):
        logger.info("Sending command: unmute")
        self.send_command({"cmd": "unmute"})

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Start the server in a background daemon thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        config = uvicorn.Config(
            self._app,
            host="127.0.0.1",
            port=self.port,
            log_level="info",
            loop="asyncio",
        )
        self._server = uvicorn.Server(config)
        self._loop.run_until_complete(self._server.serve())

    def stop(self):
        """Signal uvicorn to shut down gracefully."""
        if self._server:
            self._server.should_exit = True
