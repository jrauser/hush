# Hush — Implementation Notes

Checkpoint document recording decisions made, problems encountered, and deviations from `plan.md` during implementation.

---

## Status

| Component | Status |
|---|---|
| `label_store.py` | Done |
| `capture.py` | Done |
| `label.py` | Done |
| `train.py` | Done |
| `monitor.py` | Done |
| `mute_fsm.py` | Done |
| `tests/` | Done (36 tests) |

All components implemented. Git repo initialized with initial commit `e28ffe4`.

---

## Environment

### Python version: 3.12 (not 3.9 as originally planned)

- Python 3.9 was initially used because it had PyTorch wheels and built-in tkinter.
- Running `brew install python-tk@3.13` later in the session silently removed `tcl-tk` 8.6, which Python 3.9's `_tkinter.so` was dynamically linked against. Result: tkinter entirely broken on 3.9 with no recoverable path.
- Python 3.13/3.14 can't be used because PyTorch dropped macOS `x86_64` wheels at v2.5.0, and 3.13 wheels only exist from 2.5.0 onward — the constraints are mutually exclusive on this Intel Mac.
- **Solution:** Python 3.12 + `python-tk@3.12`, which links against tcl-tk 9.0.3. Supports PyTorch up to 2.4.1.

**Current venv:** `/Users/jrauser/code/hush/.venv` — Python 3.13.12 (arm64, via `/opt/homebrew`)

```bash
uv venv --python python3.13 .venv
uv pip install -r requirements.txt
```

### Installed package versions

```
torch==2.11.0
torchvision==0.26.0
numpy==2.4.4
Pillow==12.1.1
scikit-learn==1.8.0
scipy==1.17.1
joblib==1.5.3
tqdm==4.67.3
matplotlib==3.10.8
flask==3.1.3
tkinter (tcl/tk 9.0.3, via python-tk@3.13)
```

This machine is Apple Silicon (M5 Pro). Feature extraction runs on MPS.

### ResNet-50 weights: SSL workaround

Homebrew Python doesn't use the macOS system certificate store, so the automatic download of ResNet-50 weights (`IMAGENET1K_V2`) fails with an SSL error. Workaround: download manually with curl to the torch cache directory:

```bash
curl -L -o ~/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth \
  https://download.pytorch.org/models/resnet50-11ad3fa6.pth
```

---

## Feature Extraction: Why ResNet-50

Color histograms were considered and rejected. The failure mode is too predictable and too annoying: pre-recorded broadcast material (athlete bios, studio commentary, halftime panels, injury reports) lacks the green-field color signal and would be muted consistently. This is worse than random error because it punishes exactly the content the user wants to hear.

ResNet-50 encodes semantic content and production grammar — network overlays, lower-third graphics, the visual style of broadcast vs. advertising — and can learn to distinguish these from labeled examples. This is the right tool from the start; there is no value in prototyping histograms.

DINOv2 was discussed as a more modern alternative. Decision: start with ResNet-50 (simpler, well-understood, no external hub dependency), switch if accuracy proves insufficient. The extractor abstraction in `train.py` and `monitor.py` makes swapping straightforward.

---

## Classifier: GBM beats Random Forest

Initial results with `RandomForestClassifier` (n_estimators=300, class_weight='balanced'):
- AUC: 0.985, but ad recall only **45%** (class imbalance 7:1 content:ad dominated)

Switching to `HistGradientBoostingClassifier` (class_weight='balanced') on the same 1007 frames:
- AUC: **0.993**, ad recall **87%**, accuracy 0.965, macro F1 0.918

GBM is the default. RF remains available via `--classifier rf`.

ROC curve analysis (see `plot_roc.py`) showed AUC 0.991 for binary ad/rest detection, with a broad range of usable thresholds (0.2–0.8 all give FPR < 3%). The classifier outputs well-separated probabilities so threshold choice matters little in practice.

---

## Label Taxonomy

The original plan used a binary `ad` / `not_ad` taxonomy. This was expanded during implementation:

| Label | Key | Meaning | Included in training | Monitor behavior |
|---|---|---|---|---|
| Ad | `ad` | Commercial break | Yes | Mute |
| Content | `content` | Game/sport in progress | Yes | Unmute |
| Neither | `neither` | Something else on screen (recipe, browser tab, etc.) | Yes | No change |
| Ignore | `ignore` | Weird/corrupted frame, would add noise | No | — |

**Why `neither` is a real class:** The user explicitly watches while cooking and will tab away to recipes. Teaching the model what a tabbed-away screen looks like is cheap and prevents false mutes.

**`neither` monitor behavior:** When the classifier sees `neither`, it resets both the ad streak and the content streak to zero, then does nothing. The current mute state is held until the classifier returns to a known class. This prevents the monitor from making muting decisions when it has no information (screen showing a recipe doesn't tell you whether an ad was playing on the stream).

**Why `ignore` is excluded from training:** Corrupted frames, partial captures, or unusual system states that aren't representative of any real class.

**`n_not_ad` schema note:** The `training_runs` table has an `n_not_ad` column predating the 3-class taxonomy. It stores `n_content + n_neither` — slightly misleading but harmless since there are no constraints on how the column is interpreted.

---

## File Deletion / Privacy

An explicit deletion mechanism was added to handle accidentally capturing sensitive content (passwords, private conversations, etc.).

- **`d` key in the labeler grid:** Permanently deletes selected files from disk and removes DB records. No undo.
- **`d` key in zoom window:** Deletes current image and advances to the next.
- **Auto-purge on reload:** `_reload()` checks `os.path.isfile()` for every DB row and removes records for missing files. Files deleted from Finder or terminal are cleaned up on the next reload.

---

## `label.py` Key Decisions

**Pagination over scrolling:** Fixed pages with arrow-key navigation. Rendering hundreds of images into a scrollable canvas is slow and the scroll position is hard to track.

**Dynamic page size:** `n_cols` and `n_rows` computed from `winfo_width()` / `winfo_height()` each render cycle. `_effective_page_size = n_cols * n_rows` is updated accordingly.

**Background thread for thumbnail loading:** PIL image loading happens in a `threading.Thread`; `ImageTk.PhotoImage` construction happens in the main thread after images arrive via `queue.Queue`. The background thread sets a stop event when the page changes, so stale loads don't clobber a newly rendered page.

**PhotoImage GC pitfall:** All `ImageTk.PhotoImage` objects are stored in `self._photo_refs`. If this list is not maintained, Python's GC collects them and thumbnails go blank with no error message.

**`_text_widget_focused()` guard:** Single-key shortcuts (`a`, `c`, `n`, `x`, `d`, `z`) are suppressed when a text input widget has focus, checked via `winfo_class()`.

**Fast label update:** `_label_selected()` updates `_all_rows` in-memory rather than calling `_reload()`. This keeps labeling instant even with large datasets.

**Undo does not cover deletes.** The undo stack tracks label changes only.

---

## `capture.py` Key Decisions

**`--timeout` flag (default 2 hours):** `time.monotonic()` used for the deadline (immune to clock adjustments).

**23-second interval:** Ensures at least ~2 frames per ad break (typical US ad break: ~30 seconds minimum). Prime number avoids aliasing with any fixed-interval broadcast events.

**`screencapture -x -t jpg -m`:** `-x` suppresses the camera shutter sound; `-t jpg` for smaller files; `-m` captures only the main monitor. If the stream is on a secondary display, `-m` needs to be replaced with `-D <n>`.

---

## `train.py` Key Decisions

**Feature cache:** Keyed by `(extractor_name, n_rows)` with a full path-list comparison for staleness. Re-running `train.py` with the same labeled set skips extraction entirely.

**predict_proba mapping:** `clf.classes_` may not be `[0,1,2]` if a fold happens to lack a class. The code maps each column through `clf.classes_` to a full `[N, 3]` probability matrix before computing metrics.

**Model bundle:** Serialized with `joblib.dump` as a dict containing `classifier`, `extractor` name, `label_map`, `label_names`, `trained_at`, `n_training_samples`. The map is included so `monitor.py` doesn't need to hardcode integer-to-label mappings.

---

## `monitor.py` Key Decisions

### Architecture

- **Main thread:** tkinter window showing rolling frame history + current mute state
- **Background thread:** capture → classify loop; sends `FrameEntry` objects to main thread via `queue.Queue`
- `root.after(250, _poll_queue)` drains the queue and updates the GUI every 250 ms

### Pause/resume button

A **⏸ Pause / ▶ Resume** toggle button freezes both capture and muting. The intended workflow: pause, inspect and correct frames, resume, switch back to the stream.

Auto-pausing based on OS-level window focus was attempted but proved unreliable on macOS. Both `focus_displayof()` and `<FocusIn>`/`<FocusOut>` event bindings failed to correctly track application-level focus — tkinter's internal focus model diverges from the OS model after `root.focus_set()` is called. The explicit button is simpler and more predictable.

**Implementation note:** `tk.Button` ignores `bg`/`fg` on macOS (native Aqua appearance). A `tk.Label` with a `<Button-1>` binding is used instead — labels do respect color settings.

### Hysteresis state machine

- `MUTE_DELAY = 2` consecutive `ad` frames before muting (10 s at 5 s interval)
- `UNMUTE_DELAY = 3` consecutive `content` frames before unmuting (15 s)
- `neither` resets both streaks to zero; current mute state is held unchanged

### User override detection

Uses `osascript` to check system mute state. If we muted the system and it's no longer muted (user manually unmuted), set `_override = True` and disengage. Override clears after `UNMUTE_DELAY` consecutive content frames.

### Frame storage policy

Monitor screenshots are temporary. Stored in `tempfile.mkdtemp()` while in the rolling history deque. When a frame is evicted (scrolls off), its temp file is deleted unless `entry.relabeled` is True. Only frames the user explicitly corrects are moved to `hard_frames/` and written to the DB — disk usage stays bounded regardless of session length.

---

## Multi-sport Model Strategy

Current training data is from UFC (ESPN). For future sports (e.g. baseball):

**Recommended: combine training data across sports.** The classifier learns "commercial production aesthetic vs. broadcast production aesthetic," not what sport is being played. Ads are largely consistent across networks; content looks different (octagon vs. diamond) but both are unambiguously "not ads" to a ResNet-50 feature extractor.

If combining causes accuracy to drop (check cross-validation AUC after adding new sport data), split into per-sport models — the `--model` CLI flag already supports this.

Potential failure mode: large class imbalance at the sport level within the `content` class. `class_weight='balanced'` handles ad/content imbalance but not sport-level imbalance within content. Watch for this in the confusion matrix if accuracy degrades.

---

## v1.1: Python 3.13 + Dependency Upgrade (2026-03-29)

### Homebrew migration: x86_64 → arm64

The machine is Apple Silicon (M5 Pro) but had only the x86_64 Homebrew (`/usr/local`) carried over from a prior Intel laptop. All Python builds were x86_64 running under Rosetta, which meant PyTorch couldn't find `macosx_arm64` wheels. Installed native arm64 Homebrew at `/opt/homebrew` and added it to PATH ahead of `/usr/local`. The old x86_64 Homebrew remains in `/usr/local` — inert but harmless.

### Python 3.12 → 3.13

No code changes required. All six scripts (`capture.py`, `label.py`, `label_store.py`, `train.py`, `monitor.py`, `plot_roc.py`) work unmodified.

### Dependency version changes

| Package | Before | After |
|---------|--------|-------|
| Python | 3.12.12 (x86_64) | 3.13.12 (arm64) |
| torch | 2.2.2 | 2.11.0 |
| torchvision | 0.17.2 | 0.26.0 |
| numpy | 1.26.4 | 2.4.4 |
| scikit-learn | 1.8.0 | 1.8.0 |
| Pillow | 12.1.1 | 12.1.1 |
| matplotlib | (installed but unlisted) | 3.10.8 |

### requirements.txt changes

- `torch>=2.0,<2.5` → `torch>=2.5` (upper bound removed; Apple Silicon has wheels for all recent versions)
- `torchvision>=0.15,<0.20` → `torchvision>=0.20`
- `numpy>=1.24,<2` → `numpy>=2.0` (torch 2.5+ supports numpy 2.x)
- Added `matplotlib>=3.8` (was missing; used by `plot_roc.py`)

### Model and feature cache compatibility

The existing `models/classifier.joblib` (trained under numpy 1.26.4) loaded successfully under numpy 2.4.4. The feature cache (`feature_cache/`) also loaded without issues. Retraining produced comparable CV metrics (AUC 0.986 vs. prior 0.993 — within normal fold variance).

---

## Package Layout and MuteFSM Extraction (v1.1)

### src/ layout

Moved all source into `src/hush/` with `pyproject.toml`. The restructure required no code changes beyond updating `from label_store import` → `from hush.label_store import` in four files. Everything else — tkinter, subprocess calls, relative file paths for screenshots/models/DB — worked as-is because those paths are resolved at runtime from the working directory, not relative to the module.

### MuteFSM: returns actions, doesn't perform them

The mute state machine was extracted from `MonitorApp` into a standalone class that takes `(pred_label, user_unmuted)` and returns `'mute'`, `'unmute'`, or `None`. It never calls `osascript`, never checks system volume, never touches tkinter. `MonitorApp._on_new_frame()` does the override detection (reading system mute state) and passes the result in, then acts on whatever the FSM returns.

This made the FSM fully testable with zero mocks — 15 tests covering all edge cases run in <1ms.

One minor wart: `MonitorApp` accesses `self._fsm._paused` (a private attribute) to check pause state in the capture loop and header rendering. Adding a `@property` felt like ceremony for an internal collaborator, but if `MuteFSM` ever gets used elsewhere this should be cleaned up.

---

## Feature Vectors in SQLite (v1.1)

Replaced the flat-file feature cache (`feature_cache/`) with a `features` table in the existing SQLite database. The old cache was keyed by `(extractor_name, sample_count)` with a full path-list comparison — adding or removing one label changed the count and invalidated the entire cache, forcing re-extraction of all images.

### Schema

```sql
CREATE TABLE features (
    screenshot_id  INTEGER NOT NULL REFERENCES screenshots(id),
    extractor      TEXT NOT NULL,
    vector         BLOB NOT NULL,
    PRIMARY KEY (screenshot_id, extractor)
);
```

### Design decisions

**BLOB over `.npy` format.** Vectors are stored as raw bytes via `ndarray.tobytes()` / `np.frombuffer()`. No numpy version-specific headers, no file format to parse. 8,192 bytes per row (2048 × float32).

**No automatic migration.** First run after the change extracted all 1,177 images from scratch (~25s on MPS) rather than parsing old cache files. The old cache was already fragile, so there was nothing worth preserving.

**INSERT OR REPLACE for upserts.** If a feature vector is re-extracted (e.g. after switching extractor versions), the new vector silently replaces the old one via the composite primary key.

**No FK enforcement.** SQLite doesn't enforce `REFERENCES` without `PRAGMA foreign_keys = ON`. Orphan feature rows from deleted screenshots waste negligible space and are invisible to queries (which always filter by requested IDs).

### CV metrics after migration

Comparable to prior run (AUC 0.983 vs. 0.986) — within normal fold variance. The feature vectors round-trip through BLOB serialization without loss.

---

## Audio Detection: Not Pursued

- **Pros:** Crowd noise is a strong signal; no GPU needed; works with screen covered.
- **Cons:** Some ads use stadium sounds; halftime/bios have music similar to ads; macOS audio capture requires a virtual driver (BlackHole/Loopback) — significant setup friction.
- **Conclusion:** Video classification is the right primary approach. Audio as a secondary confirmation signal is worth revisiting after the video classifier is validated in real use.

---

## v2.0: Chrome Extension + PyQt6 Desktop App (2026-03-30)

### Architecture

v2 replaces the v1 collection of scripts with two components:

1. **Chrome extension** — uses `chrome.tabCapture.getMediaStreamId()` + an offscreen document to grab frames from a single browser tab in the background. Posts JPEG frames to the desktop app over HTTP, receives mute/unmute commands over WebSocket. Works while the user is in other tabs or apps.
2. **PyQt6 desktop app** — unified Live Monitor view. Runs a FastAPI+uvicorn server on `localhost:9099` (single port for both HTTP frame upload at `/frame` and WebSocket control plane at `/ws`). Classifies frames, drives the MuteFSM, and sends mute/unmute commands back to the extension.

This replaces v1's `screencapture` + `osascript` approach. Audio control is now Chrome-only (tab-level muting via `chrome.tabs.update`), eliminating the system-wide mute that affected all apps.

### New files

| File | Purpose |
|------|---------|
| `extension/manifest.json` | Chrome Manifest V3 extension config |
| `extension/background.js` | Service worker: WebSocket, offscreen lifecycle, tab mute, backoff |
| `extension/offscreen.html/js` | Offscreen document: getUserMedia, frame capture loop, POST to server |
| `extension/popup.html/js` | Extension popup: status display, Enable/Disable, unmute override |
| `extension/icons/` | Green (connected) and grey (disconnected) icons |
| `src/hush/server.py` | FastAPI server: `/frame` endpoint, `/ws` WebSocket, signal bridge |
| `src/hush/frame_pipeline.py` | Classify frames, drive FSM, manage temp files |
| `src/hush/v2app.py` | PyQt6 desktop app with Live Monitor view |
| `tests/test_server.py` | FastAPI endpoint tests (11 tests) |
| `tests/test_resilience.py` | Backoff, disconnection, concurrent ops (12 tests) |
| `tests/test_frame_pipeline.py` | Pipeline with fake classifier (11 tests) |
| `tests/test_model_loading.py` | Model validation (5 tests) |

### MuteFSM: user-muted override

The FSM gained a `user_muted` parameter to handle the case where the user manually mutes the tab (via Chrome's speaker icon or the extension popup). Two override modes now exist:

- **`override_unmuted`** (existing): user manually unmuted while we had it muted. Hands off until a clean content run clears the override.
- **`override_muted`** (new): user manually muted while we had it unmuted. Hands off — no unmuting even if the classifier sees content. Clears only when the user manually unmutes (not on a content run), because the user's intent is silence.

The `_override_is_muted` flag tracks which type of override is active. 19 FSM tests now (was 15).

### Dependency changes

| Package | v1.1 | v2.0 | Notes |
|---------|------|------|-------|
| flask | `>=3.0` | removed | Replaced by FastAPI |
| fastapi | -- | `>=0.110` | HTTP + WebSocket on one port |
| uvicorn[standard] | -- | `>=0.27` | ASGI server for FastAPI (standard extra includes websockets) |
| python-multipart | -- | `>=0.0.6` | Required by FastAPI for file uploads |
| PyQt6 | -- | `>=6.6` | Desktop app GUI |

### Design decisions

**FastAPI over Flask.** Flask has no native WebSocket support — we'd have needed a separate `websockets` server on a second port. FastAPI (via Starlette) serves HTTP and WebSocket on one port, one thread, one event loop. The extension connects to a single `localhost:9099` and doesn't need to manage two endpoints.

**Signal bridge pattern.** The FastAPI server runs in a daemon thread with its own asyncio event loop. GUI updates must happen on the main (Qt) thread. A `SignalBridge` QObject emits Qt signals from the server callbacks, which Qt marshals to the main thread via its event loop. This avoids any direct cross-thread GUI access.

**Frame pipeline as a separate module.** The classify → FSM → action chain was extracted from the app into `frame_pipeline.py` so it can be tested with a fake classifier (no real model, no GPU). The pipeline receives raw bytes, writes to a temp file, classifies, and returns a `FrameResult` with the FSM action. The app acts on whatever the pipeline says.

**`tabCapture` over `captureVisibleTab`.** The extension must capture frames even when the user is in another tab or app. `tabCapture` provides a persistent MediaStream from the monitored tab. The tradeoff is Chrome shows a "sharing" indicator, but this is unavoidable for background capture.

**Extension backoff.** Exponential backoff on WebSocket reconnection: 1s → 2s → 4s → ... → 30s (cap). Resets to 1s on successful connection. Frame POST failures are drop-and-continue (no queuing) — the next capture proceeds on schedule. On disconnect, the extension stops capturing and unmutes the tab.

### Manifest V3: offscreen document for tabCapture

The initial implementation called `chrome.tabCapture.capture()` directly from the service worker. This silently fails in MV3 — the service worker lacks DOM/media APIs and `capture()` requires a user gesture context that doesn't propagate through message passing.

**Solution: three-part architecture.**
1. **Popup** calls `chrome.tabCapture.getMediaStreamId({targetTabId})` on the Enable button click (has user gesture context). Passes the stream ID to the service worker.
2. **Service worker** creates an offscreen document via `chrome.offscreen.createDocument()` with reason `USER_MEDIA`, sends it the stream ID.
3. **Offscreen document** calls `navigator.mediaDevices.getUserMedia()` with `chromeMediaSource: 'tab'` and the stream ID. Handles the frame capture loop (ImageCapture → OffscreenCanvas → JPEG blob) and POSTs directly to the server.

The service worker stays focused on WebSocket management, tab mute/unmute, and offscreen document lifecycle. The manifest already had the `offscreen` permission; `host_permissions` for `http://localhost:9099/*` was added for the fetch/WebSocket calls.

**Override detection** was also improved: replaced the polling approach (`checkMuteOverride()` before each frame POST) with `chrome.tabs.onUpdated` listener filtered by `reason === "user"`. This catches user-initiated mute toggles immediately and distinguishes them from our own programmatic muting (which has `reason === "extension"`).

### Extension popup: Enable button and auto-refresh

The initial popup was read-only status display. The user had no way to initiate a connection — the service worker auto-connected on load, but MV3 service workers are terminated after inactivity, so the connection attempt might never happen.

**Enable/Disable button** in the popup is now the primary control. "Enable on this tab" identifies the current tab, obtains the capture stream ID, and tells the service worker to connect. "Disable" tears everything down. Auto-connect on service worker startup was removed.

**Auto-refresh** via `setInterval(refreshState, 1000)` so the popup reflects state changes (connecting → connected, idle → capturing) without reopening.

### Uvicorn WebSocket support

`uvicorn` without extras does not include a WebSocket library. The initial `pyproject.toml` listed `uvicorn>=0.27`, which installed the minimal package. WebSocket upgrade requests returned 404 with "No supported WebSocket library detected." Fixed by changing the dependency to `uvicorn[standard]>=0.27`, which pulls in `websockets`.

### Graceful shutdown

Qt's event loop swallows SIGINT, so Ctrl+C had no effect. Fixed with `signal.signal(signal.SIGINT, lambda *_: app.quit())` plus a 200ms QTimer that fires a no-op callback — this forces periodic returns from Qt's C++ event loop to Python so the signal handler can run.

Uvicorn shutdown was also broken: `loop.call_soon_threadsafe(loop.stop)` yanked the event loop out from under uvicorn, causing `RuntimeError: Event loop stopped before Future completed`. Fixed by setting `server.should_exit = True` instead, which lets uvicorn finish its shutdown sequence cleanly.

### Monitor grid: fill-right-to-left with animated slide-down

The initial grid rendered newest-first at top-left, pushing all existing thumbnails right and wrapping. Every frame arrival caused every thumbnail to shift sideways — disorienting and hard to track.

**New layout:** Frames are always in chronological order left-to-right, top-to-bottom. The top row fills right-to-left: empty cells get dashed-border placeholders, and each new frame replaces the rightmost placeholder. Nothing moves sideways. A new placeholder row is created only when the first frame that needs it arrives (not when the last cell fills), so the frame always lands where the user expects.

When a new row IS created, all frame widgets animate downward (300ms ease-out-cubic via `QPropertyAnimation` on the `pos` property). This required switching from `QGridLayout` to manual positioning inside a `QScrollArea` — layouts fight with position animation, but manual `widget.move()` gives full control.

### Test coverage

81 tests total (was 36). New test files cover:
- Server endpoints: frame upload, WebSocket connect/disconnect, heartbeats, overrides, malformed input
- Resilience: callback exceptions don't crash the server, rapid POSTs, concurrent frame+WebSocket, disconnection cleanup
- Backoff schedule: doubling, cap at 30s, reset on success
- Frame pipeline: classification, uncertainty detection, temp file lifecycle, relabeling persistence
- Model loading: valid/missing/corrupt/incomplete model bundles

### What's deferred to v2.1

- Capture & Label view (v1 `label.py` still works)
- In-app training with diagnostics
- Model browser with ROC curves
