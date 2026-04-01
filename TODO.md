# Hush — TODO


## Question

If I train and pick a new model during live capture is it swapped in and used for inference right away or does it require a restart?

## BUG report - in live monitor mode right to left grid fill breaks after running for a while.

When you start up the app and start live monitoring, the frames fill in exactly like we want.  newest at the top, placeholders pop in, frames fill right to left and the old rows animate down as the new row pops in. 

But after a while (maybe once we've captured enough frames that we start to purge old ones??  I'm not sure what the trigger is.) the desired behavior breaks.  New frames always appear in the same spot pushing all other frames to the right/down.  This makes selecting difficult as the frames are moving targets.


## BUG report - extension gets wedged it left on while app is missing for a long time

Need to characterize the bug with enough detail that it can be fixed.


## Background tab capture for audio-only monitoring

Use case: you're working on something else while a sports stream plays in a background Chrome tab. You can hear the audio — your brain handles "is something exciting happening?" — but you want ads muted automatically.

The current `screencapture` approach grabs whatever is on screen, so it captures your work, not the stream. We need the stream's video frames even when the tab isn't visible.

**Approach:** A small Chrome extension using the `chrome.tabCapture` API.

- The extension captures video (and optionally audio) from the stream's tab regardless of foreground state
- Periodically sends frames to a local endpoint (HTTP POST to the monitor)
- Monitor responds to each POST with config (including capture interval), so the monitor drives timing without the extension needing independent settings. Extension defaults to a sensible interval (e.g. 5s) if it can't reach the monitor.
- Monitor feeds those frames into the existing ResNet classifier — no new model needed
- Audio-based features (volume spikes, crowd noise) could be a useful second signal but aren't required for the initial version
- User activates capture by clicking the extension's toolbar icon while on the streaming tab. This triggers `chrome.tabCapture.capture()` (which requires a user gesture). Clicking again stops the stream. Icon reflects active/inactive state. Capture is tied to the tab — survives tab switches but ends if the tab is closed.
- Extension itself would be ~50 lines of JS

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

## Store feature vectors in SQLite (v1.1, 2026-03-29)

Replaced flat-file feature cache (`feature_cache/`) with a `features` table in SQLite. Feature extraction is now incremental — labeling one new image extracts one vector, not all 1,177. Multiple extractors coexist naturally via composite PK `(screenshot_id, extractor)`.

## Upgrade to Python 3.13 + modern dependencies (v1.1, 2026-03-29)

Migrated to arm64 Homebrew (`/opt/homebrew`), Python 3.13.12, torch 2.11.0, numpy 2.4.4. See `implementation_notes.md` for details.

## Unit tests + package layout (v1.1, 2026-03-29)

Reorganized into `src/hush/` package with `pyproject.toml`. Extracted `MuteFSM` from `monitor.py`. 29 tests (label_store + mute_fsm), zero mocks.
