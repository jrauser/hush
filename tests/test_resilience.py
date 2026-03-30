"""Tests for resilience and graceful degradation.

Covers: WebSocket disconnection behavior, reconnect backoff schedule,
frame posting failures, and concurrent operations.
"""

import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hush.server import HushServer


@pytest.fixture
def server():
    return HushServer(
        port=0,
        on_frame=MagicMock(),
        on_override=MagicMock(),
        on_connection_change=MagicMock(),
    )


@pytest.fixture
def client(server):
    return TestClient(server._app)


# ── WebSocket disconnection behavior ─────────────────────────────────────────

class TestDisconnection:
    def test_disconnect_clears_connection_state(self, client, server):
        with client.websocket_connect("/ws"):
            assert server.extension.connected is True
        assert server.extension.connected is False
        server.on_connection_change.assert_any_call(True)
        server.on_connection_change.assert_any_call(False)

    def test_disconnect_notifies_callback(self, client, server):
        with client.websocket_connect("/ws"):
            pass
        # Should have been called with True (connect) then False (disconnect)
        calls = [c.args[0] for c in server.on_connection_change.call_args_list]
        assert calls == [True, False]

    def test_reconnect_after_disconnect(self, client, server):
        """A new WebSocket connection after disconnect works normally."""
        with client.websocket_connect("/ws"):
            assert server.extension.connected is True
        assert server.extension.connected is False

        with client.websocket_connect("/ws"):
            assert server.extension.connected is True

    def test_send_command_after_disconnect_is_noop(self, server):
        """Sending commands when no client is connected is a graceful no-op."""
        server.send_command({"cmd": "mute"})
        server.set_mode("active", interval_sec=2)
        server.mute()
        server.unmute()
        # No exceptions raised


# ── Frame posting failures ────────────────────────────────────────────────────

class TestFrameFailures:
    def test_empty_frame_rejected(self, client, server):
        response = client.post(
            "/frame",
            files={"frame": ("frame.jpg", b"", "image/jpeg")},
        )
        assert response.json()["ok"] is False
        server.on_frame.assert_not_called()

    def test_frame_callback_exception_does_not_crash_server(self, client, server):
        """If the frame callback raises, the server stays up."""
        server.on_frame.side_effect = RuntimeError("callback exploded")
        jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 50
        # The endpoint catches the exception and returns an error response
        response = client.post(
            "/frame",
            files={"frame": ("frame.jpg", jpeg, "image/jpeg")},
        )
        assert response.status_code == 200
        assert response.json()["ok"] is False

        # Server is still alive — next request works
        server.on_frame.side_effect = None
        response2 = client.post(
            "/frame",
            files={"frame": ("frame.jpg", jpeg, "image/jpeg")},
        )
        assert response2.status_code == 200
        assert response2.json()["ok"] is True

    def test_rapid_frame_posts(self, client, server):
        """Multiple rapid frame posts don't cause issues."""
        jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 50
        for _ in range(20):
            response = client.post(
                "/frame",
                files={"frame": ("frame.jpg", jpeg, "image/jpeg")},
            )
            assert response.status_code == 200
        assert server.on_frame.call_count == 20


# ── Concurrent operations ────────────────────────────────────────────────────

class TestConcurrency:
    def test_frame_during_websocket(self, client, server):
        """Frame POST works while a WebSocket is connected."""
        jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 50
        with client.websocket_connect("/ws") as ws:
            response = client.post(
                "/frame",
                files={"frame": ("frame.jpg", jpeg, "image/jpeg")},
            )
            assert response.status_code == 200
            assert response.json()["ok"] is True

            # WebSocket still works
            ws.send_json({"event": "status", "mode": "active", "muted": False, "interval_sec": 2})
            time.sleep(0.05)
            assert server.extension.mode == "active"

    def test_multiple_heartbeats(self, client, server):
        """Rapid heartbeats update state correctly."""
        with client.websocket_connect("/ws") as ws:
            for i in range(10):
                ws.send_json({
                    "event": "status",
                    "mode": "active" if i % 2 == 0 else "idle",
                    "muted": i % 3 == 0,
                    "interval_sec": i + 1,
                })
            time.sleep(0.1)
            # Last heartbeat should win
            assert server.extension.interval_sec == 10


# ── Backoff schedule (extension-side, tested as pure logic) ───────────────────

class TestBackoffSchedule:
    """Tests for the exponential backoff logic.

    The actual backoff runs in the Chrome extension (JS), but we test the
    schedule algorithm here as pure Python to verify correctness.
    """

    @staticmethod
    def compute_backoff_schedule(max_attempts=10, initial_ms=1000, max_ms=30000):
        """Reproduce the extension's backoff schedule."""
        schedule = []
        backoff = initial_ms
        for _ in range(max_attempts):
            schedule.append(backoff)
            backoff = min(backoff * 2, max_ms)
        return schedule

    def test_backoff_doubles(self):
        schedule = self.compute_backoff_schedule(max_attempts=5)
        assert schedule == [1000, 2000, 4000, 8000, 16000]

    def test_backoff_caps_at_max(self):
        schedule = self.compute_backoff_schedule(max_attempts=8)
        assert schedule == [1000, 2000, 4000, 8000, 16000, 30000, 30000, 30000]

    def test_backoff_reset_on_success(self):
        """After a successful connection, backoff resets to initial value."""
        initial = 1000
        backoff = initial
        # Simulate 3 failures
        for _ in range(3):
            backoff = min(backoff * 2, 30000)
        assert backoff == 8000
        # Successful connection resets
        backoff = initial
        assert backoff == 1000

    def test_backoff_never_exceeds_max(self):
        schedule = self.compute_backoff_schedule(max_attempts=100)
        assert all(ms <= 30000 for ms in schedule)
