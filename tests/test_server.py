"""Tests for the FastAPI server (HTTP + WebSocket endpoints)."""

import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hush.server import HushServer, FrameData


@pytest.fixture
def server():
    """Create a HushServer with mock callbacks."""
    return HushServer(
        port=0,  # not actually binding
        on_frame=MagicMock(),
        on_override=MagicMock(),
        on_connection_change=MagicMock(),
    )


@pytest.fixture
def client(server):
    """FastAPI TestClient wrapping the server's app."""
    return TestClient(server._app)


# ── Frame endpoint ────────────────────────────────────────────────────────────

class TestFrameEndpoint:
    def test_post_frame_valid_jpeg(self, client, server):
        # A minimal valid JPEG (just the header bytes — enough for the endpoint)
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        response = client.post(
            "/frame",
            files={"frame": ("frame.jpg", jpeg_bytes, "image/jpeg")},
            data={"timestamp": "2026-03-30T10:00:00", "tab_url": "https://example.com"},
        )
        assert response.status_code == 200
        assert response.json()["ok"] is True
        server.on_frame.assert_called_once()
        frame_data = server.on_frame.call_args[0][0]
        assert isinstance(frame_data, FrameData)
        assert frame_data.timestamp == "2026-03-30T10:00:00"
        assert frame_data.tab_url == "https://example.com"
        assert frame_data.image_bytes == jpeg_bytes

    def test_post_frame_missing_image(self, client, server):
        response = client.post("/frame", data={"timestamp": "now"})
        assert response.status_code == 422
        server.on_frame.assert_not_called()

    def test_post_frame_empty_image(self, client, server):
        response = client.post(
            "/frame",
            files={"frame": ("frame.jpg", b"", "image/jpeg")},
        )
        assert response.status_code == 200
        assert response.json()["ok"] is False
        server.on_frame.assert_not_called()

    def test_post_frame_metadata_optional(self, client, server):
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 50
        response = client.post(
            "/frame",
            files={"frame": ("frame.jpg", jpeg_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        assert response.json()["ok"] is True
        frame_data = server.on_frame.call_args[0][0]
        assert frame_data.timestamp == ""
        assert frame_data.tab_url == ""


# ── WebSocket endpoint ────────────────────────────────────────────────────────

class TestWebSocket:
    def test_ws_connect_disconnect(self, client, server):
        with client.websocket_connect("/ws") as _ws:
            assert server.extension.connected is True
            server.on_connection_change.assert_called_with(True)
        # After context exit, connection is closed
        assert server.extension.connected is False

    def test_ws_receive_heartbeat(self, client, server):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({
                "event": "status",
                "mode": "active",
                "muted": True,
                "interval_sec": 5,
            })
            # Give the server a moment to process
            time.sleep(0.05)
            assert server.extension.mode == "active"
            assert server.extension.muted is True
            assert server.extension.interval_sec == 5

    def test_ws_receive_override_unmuted(self, client, server):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"event": "override", "muted": False})
            time.sleep(0.05)
            server.on_override.assert_called_with(False)
            assert server.extension.muted is False

    def test_ws_receive_override_muted(self, client, server):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"event": "override", "muted": True})
            time.sleep(0.05)
            server.on_override.assert_called_with(True)
            assert server.extension.muted is True

    def test_ws_malformed_json(self, client, server):
        """Malformed JSON should not crash the server."""
        with client.websocket_connect("/ws") as ws:
            ws.send_text("not valid json {{{")
            # Send a valid message after — connection should still be alive
            ws.send_json({"event": "status", "mode": "idle", "muted": False, "interval_sec": 2})
            time.sleep(0.05)
            assert server.extension.mode == "idle"

    def test_ws_unknown_event(self, client, server):
        """Unknown events should be logged but not crash."""
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"event": "unknown_thing", "data": 42})
            time.sleep(0.05)
            # Connection should still be alive
            ws.send_json({"event": "status", "mode": "active", "muted": False, "interval_sec": 2})
            time.sleep(0.05)
            assert server.extension.mode == "active"

    def test_ws_no_client_send_command(self, server):
        """Sending a command with no client connected should not crash."""
        server.send_command({"cmd": "mute"})
        server.set_mode("active", interval_sec=2)
        server.mute()
        server.unmute()
        # No exception raised — graceful no-op
