# Hush v2.0 Plan

**Status:** COMPLETE through v2.0.  Further work superceded by plan_v2.1.md.

---

## Overview

v2 replaces the v1 collection of scripts (`capture.py`, `label.py`, `train.py`, `monitor.py`) with two components:

1. **Chrome extension** -- captures frames from a single browser tab via `chrome.tabCapture` and mutes/unmutes that tab's audio on command. Works in the background so the user can use other tabs and apps freely.
2. **Desktop app** -- unified PyQt6 application for bulk capture/labeling, live monitoring, and model management. Communicates with the extension via a single FastAPI server (HTTP + WebSocket on one port).

v2 is Chrome-only. System-level audio muting via `osascript` is dropped.

### Phasing

**v2.0 (MVP):**
- Chrome extension (capture + mute/unmute + WebSocket)
- Desktop app with Live Monitor view (primary use case)
- FastAPI server (`/frame` + `/ws`)
- Basic model selection (pick from existing models on disk)

**v2.1:**
- Capture & Label view (v1 `label.py` remains usable in the meantime)
- In-app training with diagnostics
- Model browser with ROC curves

**v2.2:**
- Installation packaging (py2app / DMG)
- First-boot setup wizard
- Chrome Web Store publication

We will test end-to-end after v2.0 before proceeding.

---

## Architecture

```
 Chrome Extension                        Desktop App (PyQt6)
 +-----------------+                     +---------------------------+
 |                 |   POST /frame       |  FastAPI + uvicorn        |
 |  Tab capture    | -----------------> |    (single port, HTTP +   |
 |  (tabCapture    |                     |     WebSocket)            |
 |   API)          |                     |                           |
 |                 | <-- ws control ---  |    /frame endpoint        |
 |  Tab audio      |                     |    /ws control plane      |
 |  mute/unmute    |                     +---------------------------+
 +-----------------+                     |  Core App                 |
                                         |    Live monitor + FSM     |
                                         |    (v2.1: bulk label,     |
                                         |     training, models)     |
                                         +---------------------------+
                                         |  SQLite DB + filesystem   |
                                         +---------------------------+
```

### Communication protocols

All communication goes through the single FastAPI server on one port.

- **Frame upload:** Extension POSTs JPEG frames to `http://localhost:<port>/frame` with metadata (timestamp, tab URL) in headers or multipart fields. Default capture resolution: 800x450 (enough detail for labeling; app further resizes to 224x224 for the classifier).
- **Control plane (WebSocket):** Bidirectional WebSocket at `ws://localhost:<port>/ws`.
  - **App to extension:**
    - `{"cmd": "set_mode", "mode": "active", "interval_sec": N}` -- start capturing
    - `{"cmd": "set_mode", "mode": "idle"}` -- stop capturing (also used for "Pause")
    - `{"cmd": "mute"}` / `{"cmd": "unmute"}` -- control tab audio
  - **Extension to app:**
    - `{"event": "status", "mode": "active"|"idle", "muted": bool, "interval_sec": N}` -- periodic heartbeat (e.g. every 5s). Serves two purposes: (1) the app uses it to detect extension disconnection — if no heartbeat arrives within 2x the interval, the app considers the extension gone and updates the status bar accordingly; (2) it synchronizes state so the app's UI accurately reflects what the extension is actually doing (guards against the app and extension disagreeing after a dropped WebSocket message).
    - `{"event": "override", "muted": false}` -- user manually unmuted the tab (via Chrome speaker icon or extension popup). FSM enters override state and stops issuing mute commands until the next content run.
    - `{"event": "override", "muted": true}` -- user manually muted the tab. FSM enters override state and stops issuing unmute commands — we respect the user's choice to stay muted even if the classifier sees content.

### Port and discovery

App listens on a fixed localhost port (e.g. `localhost:9099`). Extension has this hardcoded (configurable in extension options page if needed).

---

## Component 1: Chrome Extension

### Capabilities

- Capture the monitored tab via `chrome.tabCapture.capture()`, which provides a MediaStream that works even when the tab is in the background. Periodically grab frames from the stream, encode as JPEG, POST to the app.
- Mute/unmute the tab's audio via `chrome.tabs.update(tabId, {muted: true/false})`.
- Maintain a WebSocket connection to the desktop app for control plane messages.
- Visual indicator: extension icon changes color (green = connected, grey = disconnected).

### Extension popup

The popup shows read-only status:
- Connection state (connected / disconnected)
- Current mode (idle / capturing at N-second intervals)
- Mute state (muted / unmuted)
- "Unmute" button: sends an override message to the app (same effect as clicking Chrome's tab speaker icon). The app remains the single source of truth for mute decisions.

### Resilience

- If the app is unreachable, the extension backs off exponentially on reconnect attempts (1s, 2s, 4s... up to 30s).
- If a frame POST fails or times out (3s), the frame is dropped (not queued). The next capture proceeds on schedule.
- On WebSocket disconnect, the extension stops capturing and reverts tab to unmuted.

### Installation (v2.0)

Unpacked extension loaded via `chrome://extensions` in developer mode. Chrome Web Store publication deferred to v2.2.

---

## Component 2: Desktop App

### Tech stack

- **GUI:** PyQt6
- **Server:** FastAPI + uvicorn, serving both HTTP and WebSocket on a single port. Runs in a background thread with its own asyncio event loop.
- **ML pipeline:** Same as v1 (ResNet-50 feature extraction, GBM classifier, joblib serialization)
- **Database:** Same SQLite schema as v1, extended as needed

### App layout (v2.0 MVP)

The v2.0 app window focuses on the Live Monitor view. A status bar at the bottom shows: extension connection status, current mode, frame count, active model name.

Model selection (picking from existing `.joblib` files on disk) is available via a menu or toolbar dropdown.

The Capture & Label view and Models view are deferred to v2.1.

### Live Monitor (v2.0)

Same premise as v1 `monitor.py`:

- Rolling display of recent frames with color-coded borders (red=ad, green=content, grey=neither)
- Header showing state: LIVE / MUTED / OVERRIDE
- P(ad) percentage on each thumbnail
- Orange border for uncertain classifications (<80% max probability)
- Pause/Resume button: sends `set_mode: idle` / `set_mode: active` to the extension
- Click-to-select + keystroke relabeling for corrections (A/C/N/X)
- Capture interval default: 2 seconds

**Hysteresis FSM:** Same `MuteFSM` logic from v1, but instead of calling `osascript`, it sends `mute`/`unmute` commands over the WebSocket to the extension.

**Frame storage policy:** Same as v1 -- frames are temporary unless relabeled, in which case they're persisted to disk and DB.

**Override detection:** If the user manually unmutes the tab (via Chrome's speaker icon or the extension popup's Unmute button), the extension sends an `override` event. The FSM enters override state as in v1.

### Settings (v2.1)

Capture interval is currently a CLI argument only (`--interval`). This needs an in-app control so the user can adjust it without restarting. Likely a small settings dialog or a toolbar widget (spinbox / dropdown). Should update the extension's capture interval in real time via `set_mode: active` with the new interval.

Other candidates for the settings UI: mute delay, unmute delay, server port.

### Capture & Label (v2.1)

This combines v1's `capture.py` and `label.py` into one view.

**Capture controls:**
- Start/Stop button to tell the extension to begin/cease capturing
- Capture interval setting (default: 7 seconds for bulk; prime number per requirements)
- Frame counter

**Labeling grid:**
- Paginated thumbnail grid (same UX as v1 `label.py`)
- Filter by label: All / Unlabeled / Ad / Content / Neither / Ignore
- Select + keystroke labeling (A/C/N/X/D)
- Shift-click range select, Ctrl/Cmd+A select all, Escape clear
- Undo (Ctrl+Z)
- Zoom on double-click
- Page navigation (arrow keys or buttons)

**Auto-refresh:** The grid auto-refreshes when new frames arrive and after any labeling action. New frames always append to the bottom of the current sort order so as not to create moving targets during selection.

### Models (v2.1)

**Training:**
- "Train New Model" button spawns a background thread
- Progress indicator during training (feature extraction + CV)
- On completion: display per-fold table (accuracy, F1, AUC), confusion matrix, ROC curve
- Optional notes field (as in v1 `--notes`)

**Model browser:**
- List of trained models (from `training_runs` table + model files on disk)
- Select a model to see its diagnostics (metrics, ROC curve, training date, sample counts)
- "Use This Model" button to make it the active model for live monitoring

---

## Label taxonomy

Unchanged from v1. The "neither" class is retained but expected to see less use since the extension captures only the stream tab (no more "tabbed away to a recipe" frames). Remaining "neither" candidates: studio segments, halftime, interstitials, buffering screens.

| Label | Meaning | Training | Monitor behavior |
|-------|---------|----------|-----------------|
| `ad` | Commercial break | Yes | Mute |
| `content` | Game / sport in progress | Yes | Unmute |
| `neither` | Ambiguous (studio, halftime, etc.) | Yes | Hold current state |
| `ignore` | Corrupted / unrepresentative | No | -- |

---

## Database changes

The v1 SQLite schema should mostly work. Potential additions:

- `screenshots.source` column: `'screencapture'` (v1) vs `'extension'` (v2), to distinguish provenance.
- `screenshots.tab_url` column: URL of the tab the frame was captured from.
- Models table expansion if needed for the model browser (v2.1).

---

## Migration / backwards compatibility

- v2 app can read the existing v1 database and model files.
- v1 screenshots (full-screen) and v2 screenshots (tab-only) will look different. Models trained on v1 data may not perform well on v2 frames. The user should expect to relabel and retrain after switching to v2 capture.
- v1 scripts remain in the codebase but are not the primary interface.

---

## Implementation order (v2.0 MVP)

1. Chrome extension (tabCapture + mute + WebSocket client + popup)
2. App skeleton: PyQt6 window with status bar, model selector
3. FastAPI server (background thread): `/frame` endpoint + `/ws` WebSocket, single port
4. Live Monitor view (port monitor UI from tkinter to PyQt6, integrate MuteFSM)
5. Integration testing (extension + app end-to-end)
6. Documentation (README update, installation instructions)

---

## Dependencies (v2.0)

Replace `flask` with `fastapi` + `uvicorn` in `pyproject.toml`. Add `PyQt6`.

```toml
dependencies = [
    "Pillow>=10.0",
    "torch>=2.5",
    "torchvision>=0.20",
    "scikit-learn>=1.3",
    "joblib>=1.3",
    "numpy>=2.0",
    "tqdm>=4.0",
    "fastapi>=0.110",
    "uvicorn>=0.27",
    "PyQt6>=6.6",
    "matplotlib>=3.8",
]
```

---

## Testing

v2 introduces a meaningful amount of non-UI logic that v1 didn't have. The strategy is: test everything that isn't widget rendering, skip UI automation (no pytest-qt click simulation, no Selenium).

### FastAPI server (`test_server.py`)

FastAPI's `TestClient` (built on httpx) tests HTTP and WebSocket endpoints in-process with no real server or network.

| Test | What it verifies |
|------|-----------------|
| `test_post_frame_valid_jpeg` | POST `/frame` with valid JPEG + metadata → 200, frame stored to disk and DB |
| `test_post_frame_missing_image` | POST `/frame` with no image data → 422 |
| `test_post_frame_corrupt_data` | POST `/frame` with non-JPEG bytes → appropriate error |
| `test_post_frame_metadata_fields` | Timestamp and tab_url round-trip correctly |
| `test_ws_connect_disconnect` | WebSocket `/ws` connect → server tracks client; disconnect → server clears client |
| `test_ws_send_command` | Server sends `set_mode`, `mute`, `unmute` → client receives valid JSON |
| `test_ws_receive_heartbeat` | Client sends status heartbeat → server updates connection state |
| `test_ws_receive_override` | Client sends override event → server notifies the app / FSM |
| `test_ws_malformed_json` | Client sends garbage → connection stays open, error logged, no crash |
| `test_ws_no_client_connected` | App attempts to send command with no extension connected → handled gracefully (no crash, appropriate status) |

### Resilience and backoff (`test_resilience.py`)

Tests for graceful degradation when the server or extension misbehaves. These test the Python-side logic; extension-side backoff is tested separately in JS if we add Jest.

| Test | What it verifies |
|------|-----------------|
| `test_frame_post_timeout_drops_frame` | Simulated slow `/frame` handler (>3s) → frame is dropped, next capture proceeds on schedule |
| `test_frame_post_failure_no_queue` | `/frame` returns 500 → frame is not queued for retry, next capture proceeds normally |
| `test_ws_reconnect_backoff_schedule` | After disconnect, reconnect attempts follow 1s, 2s, 4s, 8s, 16s, 30s, 30s, 30s... schedule |
| `test_ws_reconnect_resets_on_success` | After successful reconnect, backoff resets to 1s for the next disconnection |
| `test_ws_disconnect_reverts_mute` | Extension loses WebSocket connection → tab audio reverts to unmuted |
| `test_ws_disconnect_stops_capture` | Extension loses WebSocket connection → capture loop stops |
| `test_server_shutdown_during_capture` | Server shuts down while extension is actively capturing → extension detects disconnect, stops cleanly |
| `test_concurrent_frame_during_command` | Frame POST arrives simultaneously with a WebSocket command → both handled without deadlock or corruption |

### MuteFSM (`test_mute_fsm.py`)

Existing 15 tests carry forward. Add cases for the new override-via-extension-event path:

| Test | What it verifies |
|------|-----------------|
| `test_override_from_extension_event` | FSM receives override event (user unmuted via Chrome or extension popup) → enters override state, stops issuing mute commands |
| `test_override_from_extension_clears_after_content_run` | Override + `unmute_delay` consecutive content frames → override clears, normal behavior resumes |

### Frame processing pipeline (`test_frame_pipeline.py`)

Tests the chain from "frame bytes arrive" → "classify" → "FSM decision" → "emit command", with a fake classifier injected (returns fixed probabilities). No real model, no GPU.

| Test | What it verifies |
|------|-----------------|
| `test_frame_to_mute_decision` | Inject frame + classifier returning high P(ad) → after `mute_delay` frames, pipeline emits mute command |
| `test_frame_to_unmute_decision` | Inject frame + classifier returning low P(ad) → after `unmute_delay` frames, pipeline emits unmute command |
| `test_uncertain_frame_flagged` | Classifier returns max probability < 80% → frame marked as uncertain |
| `test_frame_not_persisted_unless_relabeled` | Frame arrives, gets classified, ages out → temp file deleted |
| `test_relabeled_frame_persisted` | Frame is relabeled by user → moved to permanent storage, recorded in DB |

### Model loading (`test_model_loading.py`)

| Test | What it verifies |
|------|-----------------|
| `test_load_valid_model` | Load a `.joblib` file with expected keys → returns classifier + metadata |
| `test_load_missing_file` | Path doesn't exist → clear error, app doesn't crash |
| `test_load_corrupt_file` | File exists but isn't valid joblib → clear error |
| `test_load_model_missing_keys` | Valid joblib but missing `classifier` or `label_map` key → clear error |

### Extension (JS, optional)

If we add a Jest test runner for the extension, the high-value targets are:

| Test | What it verifies |
|------|-----------------|
| `test_backoff_schedule` | Reconnect delays follow exponential backoff with 30s cap |
| `test_backoff_reset` | Successful connection resets backoff to 1s |
| `test_state_transitions` | idle → active → idle, muted → unmuted, connected → disconnected |
| `test_command_parsing` | Valid and invalid JSON from server handled correctly |
| `test_capture_stops_on_disconnect` | WebSocket drops → capture timer cleared |

These are cheap to write and cover the extension's most important logic without needing a browser.

### What we skip

- PyQt6 widget rendering, layout, click/keyboard interactions → manual testing
- End-to-end extension ↔ app with a real Chrome instance → manual testing after v2.0
- Visual correctness of thumbnails, color-coding, borders → manual testing

---

## Resolved questions

These were discussed during planning and are recorded here for context.

| # | Question | Decision |
|---|----------|----------|
| Q1 | Web UI vs PyQt6? | **PyQt6.** Tab management would be confusing with the stream also in a browser tab. |
| Q2 | `captureVisibleTab` vs `tabCapture`? | **`tabCapture`.** Must work in background while user is in other tabs/apps. |
| Q3 | Frame resolution? | **800x450 default.** Enough for labeling, small payloads. Can revisit if zoom feels too low-res. |
| Q4 | "Neither" class in v2? | **Keep it.** Less use expected, but still covers studio segments, halftime, etc. |
| Q5 | Multi-tab support? | **Single tab only.** |
| Q6 | Pause semantics? | **Tell extension to go idle** (`set_mode: idle`). |
| Q7 | System audio muting? | **Chrome-only for v2.** `osascript` muting dropped. |
| Q8 | Flask vs FastAPI? | **FastAPI + uvicorn.** Single server for HTTP + WebSocket. |
| Q9 | First-boot experience? | **Deferred to v2.2.** README covers basics for now. |
| Q10 | Scope / phasing? | **v2.0 = extension + live monitor. v2.1 = labeling + training. v2.2 = packaging.** |

---

## Open questions

None currently. Ready for implementation review.
