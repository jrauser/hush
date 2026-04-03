# hush

A trainable, automatic mute controller for streaming video. Hush watches a browser tab, classifies what's on screen, and mutes or unmutes the tab accordingly — no cloud, no subscriptions, no fixed rule set.

You train it on your own content by labeling a few hundred frames. It learns whatever visual distinction matters to you: ad breaks vs. live sports, talk segments vs. gameplay, commercials vs. a movie. The motivating use case is muting ad breaks during live sports streams, but the system is general.  Built and tested on macOS, but it should also work on Windows and Linux.

---

## How it works

Hush has two components that talk to each other over a local WebSocket:

- **Chrome extension** — Captures periodic still frames from a browser tab and sends them to the desktop app. Also receives mute/unmute commands and applies them to the tab's audio.
- **Desktop app** — Receives frames, stores them, and runs the classifier. Has three modes:
  - **Training Data** — Captures a large number of training examples. Watch a stream for an hour or two, then page through the captured frames and label them by hand.
  - **Train** — Extracts CLIP features from labeled frames and trains a Gradient Boosting classifier with cross-validation.
  - **Monitor** — Runs the trained classifier on incoming frames in real time and sends mute/unmute commands back to the extension. Frames are not stored unless you correct the classifier. After making corrections, retrain to improve accuracy.

---

## Getting Started

### Prerequisites

- Python 3.13 - On macOS: `brew install python@3.13`
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for environment management
- Google Chrome

### 1. Clone and run setup

```bash
git clone https://github.com/jrauser/hush.git
cd hush
./setup.sh
```

`setup.sh` creates the virtual environment, installs all dependencies, and
pre-downloads the CLIP model weights. Fair warning: the initial install is large
— PyTorch is ~2GB and the CLIP weights are ~350MB (~2.4GB total). Both are
cached after the first install. It also creates a `hush.command` file you can
double-click in Finder to launch the app.

### 2. Load the Chrome extension

1. Open `chrome://extensions` in Chrome.
2. Enable **Developer mode** (toggle in the top right).
3. Click **Load unpacked** and select the `extension/` folder in this repo.
4. The Hush icon appears in the toolbar (grey = disconnected from the desktop app).

### 3. Collect training data

Launch the app and navigate to the **Training Data** tab:

```bash
.venv/bin/python -m hush.v2app
```

- Click **Enable on this tab** in the Hush extension popup to connect the
  extension, then click **Bulk Capture** in the app to start capturing frames.
- Let it run through a representative stretch of content — including both the
  material you want muted and the material you want unmuted. A few hundred
  frames of each is enough to start.

### 4. Label your frames

In the Training Data tab, set the filter dropdown to **Unlabeled** and page through the captured frames, pressing:

| Key | Label |
|-----|-------|
| `A` | ad — mute this |
| `C` | content — unmute this |
| `N` | neither — hold current state |
| `X` | ignore — discard (blurry, transitional, etc.) |

You can select multiple frames and label them at once. Aim for at least a few
hundred labeled frames across `ad` and `content` before training.

### 5. Train a model

Open **Train Model** from the menu. Training runs cross-validation and reports
accuracy, F1, and AUC. If the metrics look good (AUC > 0.95 is a reasonable
bar), proceed. If not, label more frames — especially whatever the confusion
matrix shows the model is getting wrong.

The first training run may take a few minutes; subsequent runs are faster
because features are cached.

### 6. Monitor a live stream

Switch to the **Monitor** tab. Click **Enable on this tab** in the extension
popup on your stream tab. The app will start classifying frames and muting/
unmuting automatically.

- Green border = content (unmuted)
- Red border = ad (muted)
- Orange border = uncertain — the model isn't confident
- Click any thumbnail and press `A/C/N/X` to correct a mistake; corrected
  frames are saved for the next retraining run.

### Improving accuracy over time

Hush gets better the more you correct it. After a session, retrain with
**Train Model** to incorporate any corrections. A few sessions of active
correction usually gets the model to the point where it rarely makes mistakes.

---

## Reference

### CLI options

```
  --model FILE         Trained model (default: models/classifier.joblib)
  --db FILE            SQLite database (default: hush.db)
  --port PORT          Server port (default: 9099)
  --interval SECONDS   Capture interval in seconds (default: 2)
  --mute-delay N       Ad frames before muting (default: 2)
  --unmute-delay N     Content frames before unmuting (default: 3)
  --history-size N     Frames in live monitor rolling display (default: 40)
```

### Extension popup

Click the Hush icon to see: connection status, current mode, capture interval,
and mute state. An **Unmute** button appears when the tab is muted by Hush,
letting you override it.

### Label taxonomy

| Label | Meaning | Used in training | Monitor behavior |
|-------|---------|-----------------|-----------------|
| `ad` | Content you want muted | Yes | Mute |
| `content` | Content you want unmuted | Yes | Unmute |
| `neither` | Ambiguous (transitions, interstitials, etc.) | Yes | Hold current state |
| `ignore` | Corrupted / unrepresentative | No | — |

---

## Files

```
hush/
├── setup.sh                  One-time setup script
├── pyproject.toml
├── extension/                Chrome extension
│   ├── manifest.json
│   ├── background.js         Service worker: WebSocket, mute control
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
├── hush.db                   SQLite database — frames, labels, features, training runs (gitignored)
├── models/                   Trained classifiers (gitignored)
└── screenshots/              Captured and relabeled frames (gitignored)
```

---

## Running tests

```bash
.venv/bin/python -m pytest tests/ -v
```

149 tests covering: server endpoints, WebSocket protocol, resilience/backoff,
frame pipeline, model loading, training, bulk capture, model browser, mute FSM,
selection, and database operations.
