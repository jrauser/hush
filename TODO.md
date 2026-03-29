# Hush — TODO

## Background tab capture for audio-only monitoring

Use case: you're working on something else while a sports stream plays in a background Chrome tab. You can hear the audio — your brain handles "is something exciting happening?" — but you want ads muted automatically.

The current `screencapture` approach grabs whatever is on screen, so it captures your work, not the stream. We need the stream's video frames even when the tab isn't visible.

**Approach:** A small Chrome extension using the `chrome.tabCapture` API.

- The extension captures video (and optionally audio) from the stream's tab regardless of foreground state
- Periodically sends frames to a local endpoint (WebSocket or HTTP POST to the monitor)
- Monitor feeds those frames into the existing ResNet classifier — no new model needed
- Audio-based features (volume spikes, crowd noise) could be a useful second signal but aren't required for the initial version
- Extension itself would be ~50 lines of JS

## Store feature vectors in SQLite

The current feature cache (`feature_cache/`) uses flat files keyed by extractor name + sample count. This is fragile — adding one label invalidates the entire cache and forces re-extraction of all images. Moving features into the database enables incremental extraction and eliminates cache file management.

**Schema:** Add a `features` table with a composite primary key:

```sql
CREATE TABLE features (
    screenshot_id  INTEGER NOT NULL REFERENCES screenshots(id),
    extractor      TEXT NOT NULL,
    vector         BLOB NOT NULL,
    PRIMARY KEY (screenshot_id, extractor)
);
```

**Changes to `train.py`:**
- Query labeled screenshots that are missing a row in `features` for the current extractor
- Extract only those, insert the vectors
- Load all labeled features in one query, assemble numpy arrays, train
- Remove `feature_cache/` directory, `_cache_stem`, `load_cache`, `save_cache`

**Benefits:**
- Incremental: labeling one new image extracts one new feature vector, not all 1000+
- Multiple extractors coexist naturally (ResNet-50 and future DINOv2 side by side)
- No stale cache files to clean up
- ~8KB per row (2048 × float32), so ~8.5MB for 1000 images — trivial for SQLite

## Package for distribution via PyPI + uv

Rename the project (hush conflicts with a common shell) and publish to PyPI so users can install with `uv tool install muteads` (or whatever name we pick).

**Prerequisites:**
- Pick a name that's available on PyPI and doesn't collide with existing tools
- Reorganize into a proper package layout (`hush/` package with `__init__.py`, `pyproject.toml`, `tests/`). Currently all files are loose in the project root with library code (`label_store.py`) mixed in with commands.
- Add `[project.scripts]` entry points (e.g., `muteads capture`, `muteads monitor`, or individual commands)
- Migrate from `requirements.txt` to `pyproject.toml` dependencies

**User experience:**
```
brew install uv              # one-time
uv tool install muteads      # installs into isolated env, puts CLI on PATH
muteads monitor --web-port 5150
```

**Data directory:** Currently all paths are relative to the working directory. A packaged tool needs a fixed data home, e.g. `~/Library/Application Support/muteads/` (with `--data-dir` override). DB, models, screenshots, hard frames all live there. Pretrained model ships as package data and gets copied into the data directory on first run. Each command's `--db`, `--model`, etc. flags default to paths within that directory.

**Considerations:**
- Ships a pretrained "good enough" model, with the correction loop refining it per-user
- Heavy deps (PyTorch ~1GB) mean the initial install is large but pre-built wheels avoid compilation
- Training/labeling tools can ship in the same package — power users retrain, casual users just correct via the web UI

---

# Done

## Mobile annotation via local web UI (v1.0)

Implemented as a Flask server in `monitor.py` with `--web-port` flag.

## Upgrade to Python 3.13 + modern dependencies (v1.1, 2026-03-29)

Migrated to arm64 Homebrew (`/opt/homebrew`), Python 3.13.12, torch 2.11.0, numpy 2.4.4. See `implementation_notes.md` for details.

## Unit tests + package layout (v1.1, 2026-03-29)

Reorganized into `src/hush/` package with `pyproject.toml`. Extracted `MuteFSM` from `monitor.py`. 29 tests (label_store + mute_fsm), zero mocks.
