# Hush — Implementation Notes

Checkpoint document recording decisions made, problems encountered, and deviations from `plan.md` during implementation. Intended as a reference for resuming work after a break.

---

## Status

| Component | Status |
|---|---|
| `label_store.py` | Done |
| `capture.py` | Done |
| `label.py` | Done |
| `train.py` | Not started |
| `monitor.py` | Not started |

---

## Environment

### Python version: 3.12 (not 3.9 as originally planned)

The original plan specified Python 3.9. During setup this turned out to be unstable:

- Python 3.9 was initially used because it had PyTorch wheels and built-in tkinter.
- Running `brew install python-tk@3.13` later in the session silently removed `tcl-tk` 8.6, which Python 3.9's `_tkinter.so` was dynamically linked against. Result: tkinter entirely broken on 3.9 with no recoverable path.
- Python 3.13/3.14 can't be used because PyTorch dropped macOS `x86_64` wheels at v2.5.0, and 3.13 wheels only exist from 2.5.0 onward — the constraints are mutually exclusive on this Intel Mac.
- **Solution:** Python 3.12 + `python-tk@3.12`, which links against tcl-tk 9.0.3. Supports PyTorch up to 2.4.1.

**Current venv:** `/Users/jrauser/code/hush/.venv` — Python 3.12.12

```bash
uv venv --python python3.12 .venv
uv pip install -r requirements.txt
```

### Installed package versions

```
torch==2.2.2
torchvision==0.17.2
numpy==1.26.4        # must be 1.x; torch 2.2.x breaks with numpy 2.x
Pillow==12.1.1
scikit-learn==1.8.0
scipy==1.17.1
joblib==1.5.3
tqdm==4.67.3
tkinter (tcl/tk 9.0.3, via python-tk@3.12)
```

### requirements.txt upper bounds

```
torch>=2.0,<2.5      # 2.5+ dropped macOS x86_64 wheels
torchvision>=0.15,<0.20
numpy>=1.24,<2       # torch <2.5 requires numpy 1.x
```

---

## Feature Extraction: Why ResNet-50

Color histograms were considered and rejected. The failure mode is too predictable and too annoying: pre-recorded broadcast material (athlete bios, studio commentary, halftime panels, injury reports) lacks the green-field color signal and would be muted consistently. This is worse than random error because it punishes exactly the content the user wants to hear.

ResNet-50 encodes semantic content and production grammar — network overlays, lower-third graphics, the visual style of broadcast vs. advertising — and can learn to distinguish these from labeled examples. This is the right tool from the start; there is no value in prototyping histograms.

---

## Label Taxonomy

The original plan used a binary `ad` / `not_ad` taxonomy. This was expanded during implementation. The actual label values stored in the DB and the keys used in the code are:

| Label | Key | Meaning | Included in training |
|---|---|---|---|
| Ad | `ad` | Commercial break | Yes |
| Content | `content` | Game/sport in progress | Yes |
| Neither | `neither` | Something else on screen (recipe, browser tab, etc.) | Yes |
| Ignore | `ignore` | Weird/corrupted frame, would add noise | No |

**Why `neither` is a real class:** The user explicitly watches while cooking and will tab away to recipes. A classifier that treats "tabbed away to browser" as content or ad would mute at the wrong time. Teaching the model what a tabbed-away screen looks like is cheap (a handful of labeled frames) and prevents a real annoyance.

**Why `ignore` is excluded from training:** Corrupted frames, partial captures, or unusual system states that aren't representative of any real class. Feeding them in as any label would add noise; excluding them is cleaner.

**Impact on `train.py` (not yet written):** The multi-class labels require minor adjustments versus the originally planned binary setup:
- `f1_score(..., average='macro')` instead of `average='binary'`
- `roc_auc_score(..., multi_class='ovr', average='macro')`
- Confusion matrix is 3×3
- At monitor time: `content` and `neither` both map to "don't mute"; only `ad` triggers muting

**Note:** The `n_not_ad` column in the `training_runs` table was designed for binary classification. When implementing `train.py`, this will need updating — either rename it or store `n_content` + `n_neither` separately. The schema may need a migration.

---

## File Deletion / Privacy

A explicit deletion mechanism was added to handle accidentally capturing sensitive content (passwords, private conversations, etc.).

**Behavior:**
- **`d` key in the labeler grid:** Permanently deletes the selected files from disk (`os.remove`) and removes their DB records. No undo. Irreversible by design.
- **`d` key in zoom window:** Deletes the current image and advances to the next frame in the page, so multiple sensitive captures can be deleted in one zoom session without closing the window.
- **Auto-purge on reload:** When `_reload()` runs, it checks `os.path.isfile()` for every DB row and silently removes records for any files that no longer exist on disk. This means you can delete files from the Finder or terminal at any time; the next labeler reload cleans up the DB automatically.

**train.py implication:** Should also skip missing files (`os.path.isfile()` check before feature extraction) as a defensive measure, even though auto-purge should have handled it.

---

## `label.py` Key Decisions

**Pagination over scrolling:** 48 thumbnails per page (configurable). Rendering hundreds of images into a scrollable canvas is slow and the scroll position is hard to track. Fixed pages with arrow-key navigation are faster and more predictable for bulk labeling workflows.

**Background thread for thumbnail loading:** PIL image loading happens in a `threading.Thread`; `ImageTk.PhotoImage` construction happens in the main thread after images arrive via `queue.Queue`. This is required — tkinter is not thread-safe and `PhotoImage` must be created on the main thread. The background thread sets a stop event when the page changes, so stale loads don't clobber a newly rendered page.

**PhotoImage GC pitfall:** All `ImageTk.PhotoImage` objects are stored in `self._photo_refs`. If this list is not maintained, Python's GC collects them and thumbnails go blank with no error message. This is a classic tkinter footgun.

**`_text_widget_focused()` guard:** All single-key shortcuts (`a`, `c`, `n`, `x`, `d`, `z`) are suppressed when a text input widget has focus. Checked via `winfo_class()` returning `'Entry'`, `'TCombobox'`, or `'Text'`. Without this, typing in the search box would accidentally label images.

**Undo does not cover deletes.** The undo stack tracks label changes only. Deleting a file is irreversible; it is deliberately excluded from the undo mechanism.

---

## `capture.py` Key Decisions

**`--timeout` flag (default 2 hours):** Added after the user noted that leaving capture running indefinitely after a session would waste disk and potentially capture sensitive content. `time.monotonic()` is used for the deadline (not `datetime.now()`) to be immune to clock adjustments.

**23-second interval:** Chosen to ensure at least ~2 frames per ad break (typical US ad break: ~30 seconds minimum). Prime number avoids aliasing with any fixed-interval events in the stream.

**`screencapture -x -t jpg -m`:** `-x` suppresses the camera shutter sound; `-t jpg` for smaller files; `-m` captures only the main monitor. If the stream is on a secondary display, `-m` needs to be replaced with `-D <n>`.

---

## Audio Detection: Not Pursued

Discussed as an alternative to screenshot-based detection. Summary of the analysis:

- **Pros:** Crowd noise is a very strong signal (distinctive broadband noise floor present during play, absent during ads); no GPU needed; works with screen covered.
- **Cons:** Some ads deliberately use stadium sounds; athlete bios/halftime shows have music+voiceover similar to ads; macOS system audio capture requires a virtual audio driver (BlackHole/Loopback) plus routing setup — significant friction.
- **Conclusion:** Video classification is the right primary approach given the existing infrastructure. Audio as a secondary confirmation signal (both must agree before muting) is worth revisiting after the video classifier is validated. Not worth building now.

---

## Remaining Work

### `train.py`

See `plan.md` §6.4 for the detailed spec. Key points:
- Filter out `label == 'ignore'` rows before training
- Filter out rows where the file doesn't exist on disk
- Handle 3-class labels (`ad`, `content`, `neither`)
- Map to integers: `ad=0, content=1, neither=2` (or whatever order; just be consistent and document it in the saved model bundle)
- At monitor time `content` and `neither` → don't mute; `ad` → mute
- Feature cache keyed by labeled row count to skip re-extraction on reruns
- Save ROC data as companion `.npz` file

### `monitor.py`

**Revised Feb 28 2026:** monitor is a GUI app, not a headless daemon.

#### Architecture

- **Main thread:** tkinter window showing rolling frame history + current mute state
- **Background thread:** capture → classify loop; sends results to main thread via `queue.Queue`
- `root.after()` polling loop drains the queue and updates the GUI each tick

#### Pause/resume button

A **⏸ Pause / ▶ Resume** toggle button in the header freezes both capture and muting. While paused, the grid is static and the user can inspect and correct frames. Pressing Resume restarts the capture loop from where it left off.

Earlier attempts to auto-pause based on OS-level window focus (`focus_displayof()`, `<FocusIn>`/`<FocusOut>` events) were unreliable on macOS — tkinter's focus model doesn't always track application-level focus correctly. The explicit button is simpler and more predictable.

#### Rolling history display

- Keep a `deque(maxlen=N)` of recent frames (default: 24 frames ≈ 2 min at 5 s interval)
- Each frame record stores: `{path, label, prob_ad, captured_at}`
- Display as a horizontal strip or grid of thumbnails, newest first
- Color-coded border: red = ad, green = content/neither, yellow = uncertain (max prob < 0.80)
- Show current mute state prominently (e.g. "MUTED" in red / "LIVE" in green) in a header bar

#### In-app relabeling

- Single-click a thumbnail to select it
- a / c / n / x keys relabel the selected frame and save to DB via `store.set_label()`
- Relabeled frames are saved with `source='monitor'` so they show up in the next `label.py` / `train.py` cycle
- No undo needed (correction workflow, not bulk labeling)

#### Hysteresis state machine

- `MUTE_DELAY = 2` consecutive ad frames before muting (10 s at 5 s interval)
- `UNMUTE_DELAY = 3` consecutive non-ad frames before unmuting (15 s)
- Only `ad` class triggers muting; `content` and `neither` both count as non-ad

#### User override detection

- Record the volume level set when muting
- Before each unmute decision, check current system volume via osascript
- If it differs from what we set (user manually changed it), set a `_override` flag and stop interfering until the next mute cycle

#### Frame storage policy

Screenshots taken by the monitor are **temporary by default**:
- Stored in a temp dir (`tempfile.mkdtemp()`) while they're in the rolling history
- When a frame is evicted from the history deque (scrolls off), its temp file is deleted — **unless** the user relabeled it
- Each history entry carries a `relabeled: bool` flag; eviction checks this flag before deleting

When the user relabels a frame:
- Move the temp file to `hard_frames/` (permanent storage)
- Write a DB record via `store.add_screenshot()` + `store.set_label()` with `source='monitor'`
- Now it's training data for the next `label.py` / `train.py` cycle

Disk usage stays bounded regardless of how long the monitor runs. Only errors the user explicitly corrects are kept.

#### Other

- Unmute on clean exit (window close or SIGTERM)
- CLI: `--model`, `--db`, `--interval`, `--ad-threshold`, `--content-threshold`, `--history-size`, `--hard-frame-dir`

### Schema note for `training_runs`

The `n_not_ad` column in the DB predates the 3-class taxonomy. Options when implementing `train.py`:
1. Keep `n_not_ad` and store `n_content + n_neither` in it (simplest, slightly misleading)
2. Add `n_content` and `n_neither` columns via `ALTER TABLE` and keep `n_not_ad` for backwards compatibility
3. Store counts as a JSON blob in a new `class_counts` column

Option 1 is fine for now since there are no existing training run records.
