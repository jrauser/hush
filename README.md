# hush

Automatically mutes audio during ad breaks while streaming sports, then unmutes when the game resumes. Uses a CNN-based classifier trained on labeled screenshots.

---

## How it works

1. **Capture** frames from a browser tab via the Chrome extension.
2. **Label** frames as ad / content / neither / ignore.
3. **Train** a classifier (ResNet-50 features + Gradient Boosting).
4. **Monitor** a live stream — the classifier runs on each frame and mutes/unmutes automatically via a hysteresis state machine.

---

## Requirements

- Python 3.13
- Google Chrome
- `uv` for environment management

```bash
uv venv --python python3.13 .venv
uv pip install -e ".[dev]"
```

## Install the Chrome extension

1. Open `chrome://extensions` in Chrome.
2. Enable "Developer mode" (toggle in the top right).
3. Click "Load unpacked" and select the `extension/` directory in this repo.
4. The Hush icon appears in the toolbar (grey = disconnected).

## Run the desktop app

```bash
.venv/bin/python -m hush.v2app
```

The app starts a local server on port 9099. When the extension connects, the icon turns green. The app automatically tells the extension to start capturing.

**CLI options:**

```
  --model FILE         Trained model (default: models/classifier.joblib)
  --db FILE            SQLite database (default: hush.db)
  --port PORT          Server port (default: 9099)
  --interval SECONDS   Capture interval (default: 2)
  --mute-delay N       Ad frames before muting (default: 2)
  --unmute-delay N     Content frames before unmuting (default: 3)
  --history-size N     Frames in rolling display (default: 40)
  --hard-frame-dir DIR Where to save relabeled frames (default: hard_frames/)
```

## Using the monitor

- **Thumbnails** show classified frames with color borders (red=ad, green=content, grey=neither). Orange border = uncertain (<80% confidence).
- **Header** shows LIVE / MUTED / OVERRIDE / PAUSED / NO EXTENSION.
- **Pause/Resume** button tells the extension to stop/start capturing.
- **Correct mistakes:** click a thumbnail, press A/C/N/X to relabel. The frame is saved permanently for retraining.
- **Override:** if you manually mute/unmute the tab (via Chrome or the extension popup), the system respects your choice and stops interfering.

## Extension popup

Click the Hush icon to see: connection status, current mode, capture interval, and mute state. An "Unmute" button appears when the tab is muted.

---

## Label taxonomy

| Label | Meaning | Training | Monitor behavior |
|-------|---------|----------|-----------------|
| `ad` | Commercial break | Yes | Mute |
| `content` | Game / sport in progress | Yes | Unmute |
| `neither` | Ambiguous (studio, halftime, etc.) | Yes | Hold current state |
| `ignore` | Corrupted / unrepresentative | No | -- |

---

## Files

```
hush/
├── pyproject.toml
├── extension/                Chrome extension
│   ├── manifest.json
│   ├── background.js         Service worker: capture, WebSocket, mute
│   ├── offscreen.js          Frame capture (getUserMedia)
│   ├── popup.html / popup.js Extension popup UI
│   └── icons/                State-colored icons
├── src/hush/
│   ├── v2app.py              Desktop app (PyQt6)
│   ├── server.py             FastAPI server (HTTP + WebSocket)
│   ├── frame_pipeline.py     Frame classification pipeline
│   ├── train.py              Feature extraction + classifier training
│   ├── train_dialog.py       In-app training UI
│   ├── model_browser.py      Model browser UI
│   ├── mute_fsm.py           Mute state machine
│   ├── label_store.py        SQLite persistence layer
│   ├── selection.py          Multi-select helpers
│   └── settings_dialog.py    Settings UI
├── tests/
│   ├── test_server.py
│   ├── test_resilience.py
│   ├── test_frame_pipeline.py
│   ├── test_model_browser.py
│   ├── test_model_loading.py
│   ├── test_train.py
│   ├── test_bulk_capture.py
│   ├── test_selection.py
│   ├── test_mute_fsm.py
│   └── test_label_store.py
├── models/                   Trained classifiers
└── hard_frames/              Relabeled frames for retraining
```

---

## Running tests

```bash
.venv/bin/python -m pytest tests/ -v
```

142 tests covering: server endpoints, WebSocket protocol, resilience/backoff, frame pipeline, model loading, training, bulk capture, model browser, mute FSM, selection, and database operations.
