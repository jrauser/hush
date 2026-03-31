# Hush v2.1 Plan

**Status:** IMPLEMENTING — Phases 1–2 complete, Phase 3 next

---

## Overview

v2.1 adds three capabilities to the v2.0 desktop app:

1. **Bulk Capture & Label view** — a second mode in the main window for capturing training data and labeling frames, replacing v1's `label.py`.
2. **In-app training** — train a new model from labeled data without leaving the app, with cross-validation diagnostics.
3. **Model browser** — a separate window (from the Model menu) to inspect trained models, view metrics/ROC curves, and select the active model.

---

## Mode toggle

The main window has two modes: **Live Monitoring** and **Bulk Capture**. A toggle widget in the header bar switches between them. When switching:

- The grid is completely replaced (live frames vs. stored/labeled images).
- The header, legend, and status bar update to reflect the active mode.
- **Switching to Bulk Capture tears down live monitor state** (FSM, history deque, temp files). The FSM's hysteresis position goes stale immediately, so a fresh start on switch-back is the right default.
- **Switching to Live Monitoring stops bulk capture** if it was running. The extension starts the monitor's capture loop at the monitor interval.
- **Switching to Bulk Capture always lands in stopped/browsing state.** Capture never auto-starts — the user must click Start Capture explicitly. This avoids surprises (e.g., accidentally flooding `screenshots/` after a quick mode switch).
- Extension capture behavior follows the mode: Live mode uses the monitor interval (default 2s), Bulk Capture mode uses a bulk interval (default 7s, prime). Switching modes sends the appropriate `set_mode` command.

### Controls present in both modes

- **Pause/Resume (Live) or Start/Stop Capture (Bulk)** — same button position, different label. In Live mode it pauses/resumes the FSM and capture. In Bulk Capture it starts/stops the extension's capture loop.
- **Connection status** — DISCONNECTED indicator shows in both modes.
- **Interval control** — spinbox or similar widget in the header to adjust capture interval. Updates the extension in real time via `set_mode`.

### What changes per mode

| Element | Live Monitoring | Bulk Capture |
|---------|----------------|--------------|
| Grid contents | Rolling history (newest at top) | Paginated DB images (filtered) |
| Grid layout | Fill right-to-left, slide-down animation | Standard left-to-right, top-to-bottom |
| Header status | LIVE / MUTED / OVERRIDE / PAUSED / DISCONNECTED | CAPTURING / IDLE / DISCONNECTED + label stats |
| Controls | Pause/Resume, interval spinbox | Start/Stop Capture, interval spinbox, Filter, Search, Page nav |
| Selection | Multi-select (unified) | Multi-select (unified) |
| Relabeling keys | A/C/N/X (saves to `screenshots/`, records in DB) | A/C/N/X/D (updates DB labels, D deletes from disk) |
| Zoom | Double-click or Z (with relabel/navigate) | Double-click or Z (with relabel/delete/navigate) |
| Legend | Updated per mode | Updated per mode |

### Grid refresh in Bulk Capture mode

- **During capture:** Uses the same fill-right-to-left / slide-down-only mechanics as the live monitor grid. New frames fill placeholder cells without moving existing thumbnails. A new placeholder row is created (with slide-down animation) only when the first frame that needs it arrives. This keeps click targets stable. If the user has navigated away from the last page, the status bar updates with the new frame count but the current page is not disrupted.
- **After labeling actions:** Grid refreshes to reflect the new label state.
- **When capture is stopped:** Grid switches to standard left-to-right, top-to-bottom paginated layout (no placeholders, no animation). This is the normal browsing/labeling mode.

---

## Unified storage: drop `hard_frames/`

v1 used a separate `hard_frames/` directory for user-corrected monitor frames, distinct from `screenshots/` for bulk captures. This separation adds complexity without value — both end up in the same DB and training pipeline.

**v2.1 change:** All frames go to `screenshots/`. When the user relabels a frame in Live mode, the temp file is moved to `screenshots/` (not `hard_frames/`). The `source` column in the DB (`'monitor'` vs `'capture'`) still distinguishes provenance if needed.

**Migration:** None. The existing `hard_frames/` files stay on disk and DB paths still point to them. Training reads paths from the DB, not from a directory listing, so everything continues to work. The only code change is that `FramePipeline` writes new relabeled frames to `screenshots/` instead of `hard_frames/`, and the `--hard-frame-dir` CLI argument is removed. Over time `hard_frames/` becomes a historical artifact with a shrinking share of the training set. If it ever bothers us, a small script to move the files and update DB paths would be straightforward — but not worth doing now.

---

## Unified selection model

Both modes share identical selection behavior. This is extracted into a reusable component rather than duplicated.

### Selection affordances

- **Click:** Toggle single frame selection.
- **Shift+click:** Range select from last-clicked to current.
- **Cmd+A (Ctrl+A on non-Mac):** Select all visible frames (current page in Bulk Capture, all visible in Live).
- **Escape:** Clear selection.
- **Double-click or Z:** Open zoom window for clicked frame.

### Selection state

- `_selected: set[str]` — set of paths for selected frames.
- `_last_clicked_idx: int | None` — index of last click for range selection.
- Cleared on page change (Bulk Capture) or mode switch.

### FrameThumb changes

The existing `FrameThumb` widget is reused for both modes. For Bulk Capture, we construct `FrameResult` objects from DB rows. Fields like `pred_label` and `prob_ad` will be `None`/0 for unlabeled frames — `FrameThumb` handles this gracefully (shows the user-assigned label if present, or "unlabeled" with a neutral border).

If this makes `FrameResult` feel overloaded, we can introduce a lighter `ThumbData` dataclass that both modes use. Whichever is cleanest.

---

## Bulk Capture & Label view

### Capture controls (header area)

- **Start/Stop Capture** button: tells extension to capture at bulk interval (default 7s). Frames are saved to `screenshots/` and recorded in DB.
- **Interval spinbox:** Adjusts capture interval (1–60s range). Updates extension in real time.
- **Frame counter:** Total captured this session.

When capture is active in Bulk Capture mode and a frame arrives:
- Save to `screenshots/` with timestamp filename.
- Record in DB via `LabelStore.add_screenshot()`.
- Auto-refresh grid (see "Grid refresh" above).

### Labeling grid

- **Paginated:** Dynamic page size computed from window dimensions (same approach as v1 `label.py`).
- **Filter dropdown:** All / Unlabeled / Ad / Content / Neither / Ignore.
- **Search:** Partial filename match.
- **Page navigation:** Arrow keys or Prev/Next buttons.
- **Labeling keys:** A/C/N/X for label, D for permanent delete.
- **Undo:** Ctrl+Z undoes last label action (not deletes, matching v1 behavior).
- **Zoom:** Double-click or Z opens a zoom window with arrow-key navigation and relabel/delete.

### Data source

Bulk Capture grid reads from `LabelStore` — all screenshots regardless of source (v1 `screencapture`, v2 extension, v1 `hard_frames`, monitor corrections). This means the user's existing v1 training data is immediately available for viewing and relabeling.

---

## In-app training

### UI

A "Train Model" action in the Model menu (or a button in Bulk Capture mode's controls). Clicking it:

1. Shows a confirmation with counts: "Train on N labeled frames (X ad, Y content, Z neither)?"
2. Spawns training in a background thread.
3. Shows a progress dialog: feature extraction progress (with count, since extraction is the slow part), then "Training..." during CV + final fit.
4. On completion: opens the Model Browser window focused on the new model, showing diagnostics.

### Training pipeline

Reuses `train.py` logic but called programmatically rather than via CLI:

- Feature extraction (ResNet-50, cached in DB — only new/changed images need extraction).
- 5-fold stratified CV with HistGradientBoostingClassifier.
- Metrics: per-fold accuracy/F1/AUC, aggregate confusion matrix, ROC curve data.
- Saves model bundle to `models/` with timestamp name (e.g., `classifier_20260330_143022.joblib`).
- Saves ROC data to `models/` alongside the bundle.
- Records training run in DB.

### Training data mixing warning

If the training set contains screenshots from multiple sources (v1 full-screen + v2 tab-only), the training dialog shows a warning: "Training data includes N full-screen captures and M tab captures. These look different — check that CV metrics are acceptable." The user decides whether to proceed.

### Progress reporting

`train.py` currently prints to stdout. For in-app use, we'll add a callback mechanism:
- `on_progress(stage, current, total)` where stage is `"extracting"` or `"training"`.
- The background thread calls this; the UI updates via signal bridge.

---

## Model browser (separate window)

Opened from **Model > Browse Models...** in the menu bar. A `QDialog` or secondary window.

### Layout

- **Left panel:** List of trained models (from `training_runs` table), showing: name, date, sample count, AUC. The currently active model is highlighted.
- **Right panel:** Details for the selected model:
  - Training date, sample counts (per class), classifier type, notes.
  - Per-fold metrics table (accuracy, F1, AUC).
  - Confusion matrix (as a table or heatmap).
  - ROC curve plot (embedded matplotlib via `FigureCanvasQTAgg`).
- **"Use This Model" button:** Makes the selected model the active model for live monitoring. Updates the pipeline and status bar. **Persists the selection** to a `settings` table in the SQLite DB (or a simple config file) so future launches use this model automatically instead of defaulting to `models/classifier.joblib`.
- **"Delete Model" button:** Removes the model file and DB record (with confirmation).

### Data source

- `training_runs` table for metadata and metrics.
- `models/*.joblib` files on disk.
- `models/*_roc.npz` files for ROC curve data.
- Reconcile: if a DB record exists but the file is missing (or vice versa), show the discrepancy.

---

## Refactoring plan

### Extract shared selection logic

Currently `HushApp` has single-select only. For v2.1:

1. **`SelectionManager`** — encapsulates `_selected`, `_last_clicked_idx`, shift-click range logic, Cmd+A, Escape. Used by both modes.
2. **Grid rendering** stays in `HushApp` but dispatches to mode-specific methods: `_render_live_grid()` (existing logic) and `_render_bulk_grid()` (new, paginated from DB).

### File organization

| File | Changes |
|------|---------|
| `v2app.py` | Add mode toggle, Bulk Capture controls, dispatch grid rendering by mode. Will grow — consider splitting if it exceeds ~800 lines. |
| `frame_pipeline.py` | Minor: change relabel destination from `hard_frames/` to `screenshots/`. |
| `server.py` | Minor: Bulk Capture frame handling (save to `screenshots/` instead of temp). |
| `label_store.py` | Add `get_settings()` / `set_setting()` for persisted model selection. Possibly add paginated query method. |
| `train.py` | Add `train_model()` function callable from app (vs. `main()` CLI). Add progress callback. |
| `model_browser.py` (new) | Model browser window. |

---

## Database changes

One addition:

```sql
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

Used initially for `active_model_path`. Lightweight key-value store for app preferences that should persist across launches.

---

## Implementation order

**Phase 1: Unified selection + Bulk Capture grid (no capture, just labeling)** ✓ DONE
1. ✓ Extract selection logic into `SelectionManager` (multi-select, shift-click, Cmd+A).
2. ✓ Add mode toggle to header.
3. ✓ Build Bulk Capture grid: paginated, filtered, reads from LabelStore.
4. ✓ Port labeling keys (A/C/N/X/D), undo, zoom window.
5. ✓ Add zoom to Live mode (for consistency).
6. ✓ Tests for selection logic and grid data flow.

**Phase 2: Bulk Capture (frame ingestion)** ✓ DONE
7. ✓ Wire Bulk Capture's Start/Stop to the extension via different interval.
8. ✓ Handle incoming frames in Bulk Capture mode (save to `screenshots/`, record in DB, refresh grid).
9. ✓ Add interval spinbox to Bulk mode. (Live mode still uses `--interval` CLI arg — adding a spinbox there is low priority since the interval rarely changes during monitoring.)
10. ✓ Tests for bulk capture data flow (filtering, labeling, undo, delete, pagination).

**Phase 3: In-app training**
11. Refactor `train.py` to expose `train_model()` with progress callback.
12. Training dialog with progress and completion.
13. Training data mixing warning.
14. Tests for training pipeline integration.

**Phase 4: Model browser**
15. Add `settings` table to DB, `get_setting()`/`set_setting()` methods.
16. Build model browser window (list + detail panels).
17. Embed matplotlib for ROC curves.
18. "Use This Model" (with persistence) and "Delete Model" actions.
19. Tests for model browser data loading.

---

## Testing strategy

Same principle as v2.0: test everything that isn't widget rendering.

### New test targets

| Area | What to test |
|------|-------------|
| Selection logic | Click, shift-click range, select-all, clear, page-change reset |
| Bulk Capture data flow | Filter/paginate DB rows, label update, delete, undo |
| Capture ingestion | Frame saved to correct location, DB record created |
| Training integration | `train_model()` returns metrics, progress callbacks fire, model file created |
| Model browser data | Load training runs from DB, reconcile with files on disk, "use this model" updates pipeline |
| Settings persistence | `set_setting()`/`get_setting()` round-trip, default model on launch |

### What we skip

- Widget rendering, click coordinates, layout math → manual testing.
- Matplotlib rendering → manual testing.

---

## Dependencies

No new dependencies. `matplotlib` is already in `pyproject.toml` (used by `plot_roc.py`). The `matplotlib` Qt backend (`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`) ships with matplotlib when PyQt6 is installed.

---

## Open questions

None currently. Ready for implementation review.
