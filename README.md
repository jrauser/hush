# hush

Automatically mutes system audio during ad breaks while streaming sports, then unmutes when the game resumes. Works by periodically taking screenshots, classifying them with a trained CNN-based classifier, and controlling macOS system volume via AppleScript.

---

## How it works

**Phase 1 — Build a classifier**

1. `capture.py` takes a screenshot every 23 seconds while you watch a stream and records each frame in a SQLite database.
2. `label.py` opens a GUI where you classify frames as ad / content / neither / ignore.
3. `train.py` extracts ResNet-50 features from labeled frames and trains a Random Forest classifier.

**Phase 2 — Run in the background**

4. `monitor.py` runs continuously, classifying each new screenshot and muting/unmuting system audio based on the result.

Hard-to-classify frames from Phase 2 are saved automatically for relabeling. After labeling them, retrain and restart the monitor.

---

## Requirements

- macOS (uses `screencapture` and `osascript`)
- Python 3.12 via Homebrew with tkinter: `brew install python@3.12 python-tk@3.12`
- `uv` for environment management

```bash
uv venv --python python3.12 .venv
uv pip install -r requirements.txt
```

---

## Workflow

### Phase 1: Capture and label

```bash
# Terminal 1 — capture while watching a stream (stops automatically after 2 hours)
.venv/bin/python capture.py

# After capturing, open the labeler
.venv/bin/python label.py
```

Label enough frames to cover the variety of content you watch:
- Ad breaks (multiple advertisers, different styles)
- In-game content (field of play, replays)
- Studio segments and athlete bios (label as **content**, not ad)
- Any time you tab away to another app (label as **neither**)

Aim for at least 100–200 ad frames before training. The more variety the better.

```bash
# Train the classifier (downloads ResNet-50 weights ~100MB on first run)
.venv/bin/python train.py
```

Check the cross-validation output. AUC > 0.95 is good. If accuracy is low, label more frames and retrain.

### Phase 2: Monitor

```bash
# Run in the background while streaming
.venv/bin/python monitor.py &
```

After each streaming session, relabel any hard frames the monitor saved and retrain:

```bash
.venv/bin/python label.py    # hard frames appear automatically
.venv/bin/python train.py    # retrain with expanded dataset
# restart monitor to load the new model
```

---

## Command reference

### `capture.py`

```
.venv/bin/python capture.py [options]

  --interval SECONDS   Screenshot interval (default: 23)
  --timeout HOURS      Stop after this many hours; 0 = unlimited (default: 2)
  --output-dir DIR     Where to save screenshots (default: screenshots/)
  --db FILE            SQLite database (default: hush.db)
```

### `label.py`

```
.venv/bin/python label.py [options]

  --db FILE            SQLite database (default: hush.db)
  --thumb-size PX      Thumbnail size in pixels (default: 150)
  --page-size N        Max thumbnails per page (default: 48; actual count
                       adapts to window height)
```

**Keyboard shortcuts**

| Key | Action |
|-----|--------|
| A | Label selected as **ad** |
| C | Label selected as **content** |
| N | Label selected as **neither** (tabbed away, etc.) |
| X | Mark selected as **ignore** (corrupted frame; excluded from training) |
| D | **Permanently delete** selected files from disk |
| Cmd+A | Select all on current page |
| Shift+click | Range-select |
| Escape | Clear selection |
| Ctrl+Z | Undo last label action (delete cannot be undone) |
| Z / double-click | Open zoom window |
| ← / → | Previous / next page |

In the zoom window, A/C/N/X relabel the current image and close; D deletes and advances to the next image; ← / → navigate without closing.

Reload re-reads the database and filesystem. Any screenshot files deleted outside the app are automatically removed from the database on the next reload.

### `train.py`

```
.venv/bin/python train.py [options]

  --db FILE            SQLite database (default: hush.db)
  --model-out FILE     Where to save the model (default: models/classifier.joblib)
  --extractor NAME     Feature extractor: resnet50 (default; dinov2_vits14 planned)
  --classifier NAME    rf (Random Forest, default) or gbm (Gradient Boosting)
  --n-folds N          Cross-validation folds (default: 5)
  --notes TEXT         Free-text note recorded with this training run
```

Feature extraction results are cached in `feature_cache/`. Re-running `train.py` with the same labeled set skips extraction and goes straight to cross-validation.

Output includes a per-fold table (accuracy, F1, AUC) and a confusion matrix. ROC curve data is saved alongside the model as `models/classifier_roc.npz`.

### `monitor.py`

```
.venv/bin/python monitor.py [options]

  --model FILE         Trained model (default: models/classifier.joblib)
  --db FILE            SQLite database (default: hush.db)
  --interval SECONDS   Classification interval (default: 5)
  --mute-delay N       Consecutive ad frames before muting (default: 2 → 10s)
  --unmute-delay N     Consecutive non-ad frames before unmuting (default: 3 → 15s)
  --history-size N     Frames shown in rolling display (default: 24 ≈ 2 min)
  --hard-frame-dir DIR Where to save relabeled frames (default: hard_frames/)
```

The monitor opens a GUI window showing the last ~2 minutes of classified frames as color-coded thumbnails (red = ad, green = content, grey = neither). The header shows the current state: **LIVE**, **MUTED**, or **OVERRIDE**. Each thumbnail caption shows the predicted label and the percentage is always **P(ad)** — the probability the classifier assigned to the ad class. An orange border means the classifier was uncertain (max probability across all classes < 80%).

**Pausing:** click the **⏸ Pause** button to freeze capture and muting while you inspect frames. The button turns into **▶ Resume**. Click it again to continue. This is the intended way to switch over from your stream, correct any mistakes, and switch back.

**Correcting a misclassification:** while paused, click a thumbnail to select it (yellow border), then press A / C / N / X to relabel it. The corrected frame is moved permanently to `hard_frames/` and recorded in the database. Run `label.py` and `train.py` afterwards to incorporate the corrections into the next model.

Frames the monitor captures but you don't relabel are temporary — deleted automatically when they age out of the history window, so disk usage stays bounded.

The monitor unmutes cleanly on window close or SIGTERM. If you manually unmute while the monitor has muted, it detects the override and stops interfering until the next content run.

---

## Label taxonomy

| Label | Meaning | Included in training | Monitor behavior |
|-------|---------|---------------------|-----------------|
| `ad` | Commercial break | Yes | Mute |
| `content` | Game / sport in progress | Yes | Unmute |
| `neither` | Something else on screen (browser, recipe, etc.) | Yes | No change (hold current state) |
| `ignore` | Corrupted or unrepresentative frame | No | — |

---

## Files

```
hush/
├── capture.py           Phase 1: screenshot capture loop
├── label.py             Phase 1: labeling GUI
├── train.py             Feature extraction + classifier training
├── monitor.py           Phase 2: muting daemon with correction GUI
├── label_store.py       Shared SQLite persistence layer
├── plot_roc.py          ROC curve plotting utility
├── requirements.txt
├── hush.db              SQLite database (created at runtime)
├── screenshots/         Training screenshots
├── hard_frames/         Frames relabeled in monitor (for retraining)
├── models/
│   ├── classifier.joblib
│   └── classifier_roc.npz
└── feature_cache/       Cached ResNet-50 features (safe to delete)
```

---

## macOS permissions

`screencapture` requires **Screen Recording** permission. On first run, macOS will prompt automatically. If it doesn't, grant it in System Settings → Privacy & Security → Screen Recording for Terminal (or whichever app you use to run the scripts).
