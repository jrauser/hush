# Hush: Automatic Ad Muting System — Implementation Plan

## 1. Project Overview

Hush is a two-phase macOS system that detects streaming ads via screenshot classification and automatically mutes/unmutes audio. Phase 1 builds the data pipeline (capture, label, train). Phase 2 runs the trained classifier in real time as a background monitor.

---

## 2. Project Structure

```
hush/
├── initial_requirements.md    # Original requirements (do not edit)
├── plan.md                    # This file
├── requirements.txt
├── hush.db                    # SQLite database (created at runtime)
├── models/
│   └── classifier.joblib      # Serialized trained classifier
├── screenshots/               # Captured training frames
│   └── YYYYMMDD_HHMMSS.jpg
├── hard_frames/               # Phase 2 uncertain frames for relabeling
│   └── YYYYMMDD_HHMMSS.jpg
├── label_store.py             # SQLite persistence layer (shared)
├── capture.py                 # Phase 1: screenshot capture loop
├── label.py                   # Phase 1: tkinter GUI labeling tool
├── train.py                   # Feature extraction + classifier training
└── monitor.py                 # Phase 2: runtime mute/unmute daemon
```

**Implementation order:** `label_store.py` first (all other scripts depend on it), then `capture.py`, then `label.py`, then `train.py`, then `monitor.py`.

---

## 3. Dependencies and Tech Stack

### Python Version

Use **Python 3.9** — the system already has it at `/usr/local/bin/python3.9`, it already includes tkinter (no extra brew install needed), and it's the only installed Python version that has PyTorch wheels for Intel macOS.

**Why not 3.13/3.14?** PyTorch dropped Intel Mac (`macosx_x86_64`) wheel support at v2.5.0. Python 3.13 wheels only exist from v2.5.0 onward. These two constraints together mean PyTorch is completely unavailable for Python 3.13+ on Intel macOS — there is no version that satisfies both.

Use `uv` (already installed) to create and manage the venv:

```bash
uv venv --python python3.9 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Feature extraction strategy

**Use CNN features (ResNet-50) from the start.**

Color histograms would work well for live field-of-play content but fail systematically on pre-recorded broadcast material — athlete bios, studio commentary, halftime panels, injury updates. These segments lack the green-field color signal but are clearly *not* ads. A histogram classifier would mute them consistently, which is worse than random error because it's predictably wrong on content the user cares about.

ResNet features encode semantic content. A sports broadcast studio segment and an ad are visually distinguishable by their production grammar — network branding overlays, lower-third stat graphics, the specific visual style of broadcast production vs. advertising. A CNN can learn these cues from labeled training data; a histogram cannot.

**Training data note:** Make sure Phase 1 capture sessions include athlete bio/studio segments, and label them correctly as `not_ad`. If the training set only contains field-of-play and obvious commercials, the classifier will be blind to this case. The hard frame mechanism in Phase 2 will surface these as uncertain frames for labeling when they're first encountered.

### requirements.txt

```
Pillow>=10.0
torch>=2.0,<2.3
torchvision>=0.15,<0.18
scikit-learn>=1.3
joblib>=1.3
numpy>=1.24,<2       # torch 2.2.x requires numpy 1.x
tqdm>=4.0
```

**Rationale:**

- **Pillow**: Image loading, resizing, thumbnail generation, and conversion to `ImageTk.PhotoImage` for the labeling GUI.
- **PyTorch + torchvision**: ResNet-50 pretrained on ImageNet provides 2048-dimensional feature vectors from penultimate-layer activations. Chosen over TensorFlow for simpler dependency management on macOS.
- **numpy**: Feature matrix construction.
- **scikit-learn**: `RandomForestClassifier`; `StratifiedKFold` for cross-validation.
- **joblib**: Standard sklearn serialization. Ships with sklearn.
- **tqdm**: Progress bars for feature extraction.
- **Screenshot capture**: macOS `screencapture` via `subprocess` — no additional dependency. `-x` suppresses shutter sound; `-t jpg` produces JPEG; `-m` captures only the main monitor.

---

## 4. Resolution and Storage Analysis

### Disk usage math

A full-screen JPEG captured via `screencapture -t jpg` is approximately 200–500 KB (varies with screen content). Using ~388 KB as a working estimate:

- Capture interval: **23 seconds**
- 2-hour session: `floor(7200 / 23)` = **313 frames**
- Storage per session: `313 × 388 KB ≈ 119 MB`
- Available disk: **141 GB**
- Sessions before disk full: `141,000 MB / 119 MB ≈ 1,185 sessions` (~2,370 hours of streaming)

**Conclusion: disk usage is not a concern.** Capture at full resolution. The CNN resizes all images to 224×224 internally anyway, so higher on-disk resolution only helps if we ever want to revisit the raw data.

### Phase 2 hard frame storage

At a 5-second monitoring interval with ~5% of frames flagged as uncertain: ~36 hard frames/hour ≈ 14 MB/hour. Even at 100% retention: ~272 MB/hour. Both are fine.

---

## 5. Data Schema (SQLite)

**Database file:** `hush.db` in the project root.

### Table: `screenshots`

```sql
CREATE TABLE IF NOT EXISTS screenshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL UNIQUE,
    captured_at TEXT NOT NULL,
    label       TEXT,
    labeled_at  TEXT,
    source      TEXT NOT NULL DEFAULT 'capture'
);
-- filename: relative path, e.g. 'screenshots/20250223_143012.jpg'
-- captured_at / labeled_at: ISO 8601 strings
-- label: NULL=unlabeled, 'ad', 'not_ad'
-- source: 'capture' or 'hard_frame'

CREATE INDEX IF NOT EXISTS idx_screenshots_label  ON screenshots(label);
CREATE INDEX IF NOT EXISTS idx_screenshots_source ON screenshots(source);
```

### Table: `training_runs`

```sql
CREATE TABLE IF NOT EXISTS training_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at              TEXT NOT NULL,
    model_path          TEXT NOT NULL,
    n_folds             INTEGER NOT NULL,
    cv_accuracy_mean    REAL NOT NULL,
    cv_accuracy_std     REAL NOT NULL,
    cv_f1_mean          REAL NOT NULL,
    cv_auc_mean         REAL NOT NULL,   -- mean ROC-AUC across folds
    cv_auc_std          REAL NOT NULL,
    confusion_matrix    TEXT NOT NULL,   -- JSON: {"tp":45,"fp":3,"tn":210,"fn":8} (summed across folds)
    n_ad                INTEGER NOT NULL,
    n_not_ad            INTEGER NOT NULL,
    classifier_type     TEXT NOT NULL,
    feature_extractor   TEXT NOT NULL,   -- 'color_histogram' or 'resnet50', etc.
    notes               TEXT
);
```

The confusion matrix is stored as a JSON string (summed across all CV folds) so the training run record is self-contained. `train.py` should also save the per-fold ROC curve data (FPR/TPR arrays) to a companion `.npz` file alongside the `.joblib` model, so ROC curves can be plotted later without rerunning training. The companion file is named `<model_stem>_roc.npz`.

---

## 6. Component Implementation Details

### 6.1 `label_store.py` — SQLite Persistence Layer

Thin wrapper around `sqlite3`. Imported by all other scripts.

**Key methods:**

```python
class LabelStore:
    def __init__(self, db_path: str)                   # opens connection, creates tables
    def add_screenshot(self, filename, captured_at, source='capture') -> int
    def get_unlabeled(self) -> list[dict]              # label IS NULL
    def get_labeled(self) -> list[dict]                # label IS NOT NULL
    def set_label(self, filename, label)               # label in ('ad', 'not_ad')
    def set_labels_bulk(self, filenames, label)
    def get_all(self) -> list[dict]
    def get_by_filename(self, filename) -> dict|None   # for correction workflow
    def record_training_run(self, ...)
```

Use `conn.row_factory = sqlite3.Row` so callers access columns by name. Use `with conn:` for auto-commit/rollback. Single persistent connection per instance (no concurrent access needed).

---

### 6.2 `capture.py` — Screenshot Capture Loop

**CLI:**
```
python capture.py [--interval 23] [--output-dir screenshots] [--db hush.db]
```

**Loop:**
1. Parse args; create output directory if needed; instantiate `LabelStore`.
2. Loop indefinitely:
   - Build filename: `datetime.now().strftime('%Y%m%d_%H%M%S') + '.jpg'`
   - Run `subprocess.run(['screencapture', '-x', '-t', 'jpg', '-m', full_path])`
   - Verify file exists and is non-empty; log warning if not.
   - Call `store.add_screenshot(relative_path, captured_at=now_iso)`
   - `time.sleep(interval)`
3. Handle `KeyboardInterrupt` and `SIGTERM` gracefully; print frame count on exit.

**Constants:** `DEFAULT_INTERVAL = 23`, `DEFAULT_OUTPUT_DIR = 'screenshots'`, `DEFAULT_DB = 'hush.db'`

---

### 6.3 `label.py` — tkinter Labeling GUI

**CLI:**
```
python label.py [--db hush.db] [--screenshot-dir screenshots] [--thumb-size 150]
                [--page-size 48]
```

**Layout:**
- Top bar: filter dropdown ("All" / "Unlabeled" / "Ad" / "Not Ad") + search-by-filename field + "Go" button + status label ("Page N of M — K unlabeled total") + Prev/Next page buttons
- Center: fixed grid of thumbnail images (no scrolling within a page — pagination instead)
- Bottom bar: keyboard shortcut legend

**Pagination:**
- Default page size: 48 thumbnails (configurable via `--page-size`). This avoids rendering hundreds of images and keeps the UI snappy.
- Prev/Next buttons (also bound to Left/Right arrow keys when no thumbnail is focused) navigate pages.
- The page state resets to page 1 when the filter changes.

**Thumbnail grid:**
- Default size: 150×150 px; images letterboxed to maintain aspect ratio
- Grid fills the window width: `cols = floor(window_width / (thumb_size + padding))`
- Each cell shows: thumbnail + short filename + label (red = ad, green = not_ad, gray = unlabeled)
- Selected cells: blue 3px border

**Interactions:**
- **Left-click**: toggle selection
- **Shift+click**: range-select from last clicked to current (within the current page)
- **'a'**: label all selected as `'ad'`; clear selection; refresh current page
- **'n'**: label all selected as `'not_ad'`; clear selection; refresh current page
- **Escape**: clear selection
- **Ctrl+Z**: undo last bulk label (maintain an undo stack of `(filenames, old_labels)`)
- **Double-click**: open a zoom window (see below)
- **'z' key** while any thumbnail is focused: same as double-click — open zoom window

**Zoom window:**
- Opens a separate `tk.Toplevel` showing the full-size image scaled to fit the screen (up to ~90% of screen dimensions).
- Title bar shows filename and current label.
- Keyboard shortcuts 'a' and 'n' work in the zoom window to relabel the displayed image and close the window.
- Arrow keys in the zoom window navigate to the previous/next image in the current filtered+paginated view.

**Correction / find-by-filename workflow:**
- The search field in the top bar accepts a partial filename (e.g., `20250223_14`).
- Pressing "Go" (or Enter) filters the grid to matching filenames, regardless of the current filter dropdown.
- This is the primary way to pull up a specific frame for relabeling.
- Since labeled frames are always visible (with color coding), mislabeled frames can also be found by switching the filter to "Ad" or "Not Ad", visually scanning, clicking, and relabeling.

**Display:**
- All rows from `store.get_all()` feed the view; labeled frames show their color-coded label.
- Thumbnails load in a background thread (`threading.Thread` + `canvas.after()`) so the UI is responsive during loading.

**Implementation note:** Store all `ImageTk.PhotoImage` objects in a list — they will be garbage collected if not referenced elsewhere, causing blank thumbnails. This is a common tkinter pitfall.

Use a `tk.Frame` grid (not a scrollable canvas) since pagination eliminates the need for scrolling.

---

### 6.4 `train.py` — Feature Extraction and Classifier Training

**CLI:**
```
python train.py [--db hush.db] [--screenshot-dir screenshots]
                [--model-out models/classifier.joblib]
                [--classifier rf|gbm] [--n-folds 5]
```

**Step 1: Load labeled data**

Query `store.get_labeled()`. Map labels: `'ad'` → 1, `'not_ad'` → 0. Print class counts. Warn if ratio > 5:1.

**Step 2: Feature extraction**

```python
model = torchvision.models.resnet50(weights='IMAGENET1K_V2')
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
feature_extractor.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

ResNet-50 output after removing the final `Linear` layer: 2048-dimensional global average pool vector per image.

Process in batches of 32 via `DataLoader`. Show `tqdm` progress bar. Use MPS if available: `device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')`.

Cache the feature matrix to `features_cache.npy` + `labels_cache.npy`, keyed by the count of labeled rows in the DB. Re-running `train.py` without new labels skips extraction entirely.

**Step 3: Cross-validation**

Use `cross_validate` with manual fold iteration (rather than just passing scoring strings) so we can accumulate the full confusion matrix and ROC curve data per fold:

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                              f1_score, RocCurveDisplay)
import numpy as np

skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
fold_results = []
all_fpr, all_tpr = [], []
cm_total = np.zeros((2, 2), dtype=int)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    clf.fit(X[train_idx], y[train_idx])
    y_pred = clf.predict(X[val_idx])
    y_prob = clf.predict_proba(X[val_idx])[:, 1]
    cm = confusion_matrix(y[val_idx], y_pred)
    cm_total += cm
    auc = roc_auc_score(y[val_idx], y_prob)
    fold_results.append({
        'fold': fold,
        'accuracy': (cm[0,0]+cm[1,1]) / cm.sum(),
        'f1': f1_score(y[val_idx], y_pred),
        'auc': auc,
    })
    # store per-fold ROC curve points for later plotting
    fpr, tpr, _ = roc_curve(y[val_idx], y_prob)
    all_fpr.append(fpr)
    all_tpr.append(tpr)
```

Print a fold-by-fold table:
```
Fold | Train Acc | Val Acc | Val F1  | Val AUC
   1 |    0.998  |  0.941  | 0.937   | 0.981
   2 |    0.996  |  0.958  | 0.952   | 0.994
   ...
Mean |    0.997  |  0.951  | 0.947   | 0.989
 Std |    0.001  |  0.008  | 0.009   | 0.006

Confusion matrix (summed across folds):
              Predicted Not-Ad  Predicted Ad
Actual Not-Ad       210              3
Actual Ad             8             45
```

Save per-fold ROC data alongside the model:
```python
np.savez(model_out.replace('.joblib', '_roc.npz'),
         fpr_list=np.array(all_fpr, dtype=object),
         tpr_list=np.array(all_tpr, dtype=object))
```
This lets you plot ROC curves with a separate script (`python plot_roc.py models/classifier_roc.npz`) without rerunning training.

**Step 4: Final model and serialization**

Fit on all labeled data. Save a bundle:
```python
joblib.dump({
    'classifier':        clf,
    'feature_extractor': extractor_name,   # 'color_histogram' or 'resnet50'
    'label_map':         {'ad': 1, 'not_ad': 0},
    'trained_at':        datetime.now().isoformat(),
    'n_training_samples': len(y),
}, model_out)
```

Record the run in `training_runs` (including AUC, confusion matrix JSON, `feature_extractor='resnet50'`). Print the paths to the saved model and ROC data file.

**Classifier recommendation:** Start with `RandomForestClassifier`. It handles 2048-dim features without scaling, is robust to hyperparameters, and trains in seconds on ~300 samples. The `class_weight='balanced'` parameter handles the expected ~3:1 content/ad imbalance. If accuracy is below ~90%, try `HistGradientBoostingClassifier`.

---

### 6.5 `monitor.py` — Runtime Muting Daemon

**CLI:**
```
python monitor.py [--model models/classifier.joblib] [--interval 5]
                  [--db hush.db] [--hard-frame-dir hard_frames]
                  [--ad-threshold 0.60] [--content-threshold 0.40]
                  [--mute-delay 2] [--unmute-delay 3]
```

**Main loop:**
1. Load model bundle (`joblib.load`); reconstruct feature extractor.
2. Loop indefinitely:
   - Capture screenshot to `/tmp/hush_frame.jpg` (overwrite each cycle)
   - Extract features; get `p_ad = clf.predict_proba([features])[0][1]`
   - Pass `p_ad` to `MuteController.update()`
   - If `p_ad` is in the uncertain zone, conditionally save to `hard_frames/`
   - Log current state and probability
   - `time.sleep(interval)`
3. Handle `KeyboardInterrupt` / `SIGTERM`; unmute on exit if currently muted.

**Constants:** `DEFAULT_INTERVAL = 5`, `AD_THRESHOLD = 0.60`, `CONTENT_THRESHOLD = 0.40`, `MUTE_DELAY = 2`, `UNMUTE_DELAY = 3`, `MAX_HARD_FRAMES_PER_HOUR = 60`

---

## 7. Audio Control on macOS

Use `osascript` (AppleScript) via `subprocess`. Controls system output volume — works regardless of which browser or app is playing the stream.

```python
import subprocess

def get_volume() -> dict:
    """Returns {'output_volume': int, 'output_muted': bool}"""
    result = subprocess.run(
        ['osascript', '-e', 'get volume settings'],
        capture_output=True, text=True, check=True
    )
    # Output: "output volume:26, input volume:29, alert volume:100, output muted:false"
    parts = dict(item.strip().split(':') for item in result.stdout.strip().split(','))
    return {
        'output_volume': int(parts['output volume']),
        'output_muted': parts['output muted'].strip() == 'true'
    }

def mute():
    subprocess.run(['osascript', '-e', 'set volume output muted true'], check=True)

def unmute():
    subprocess.run(['osascript', '-e', 'set volume output muted false'], check=True)
```

**Volume tracking for user override detection:** When the monitor mutes, record `muted_at_volume = get_volume()['output_volume']`. Before unmuting, check: if `current_volume != muted_at_volume`, the user manually changed the volume — stop interfering, clear state, resume normal monitoring without taking audio actions.

---

## 8. Hysteresis and Confidence Thresholding

### State machine

The monitor tracks a state (`CONTENT` or `AD`) and consecutive-frame counters. The gap between `content_threshold` (0.40) and `ad_threshold` (0.60) is an "uncertain zone" — frames landing there decay both counters but don't increment either. This handles commentary segments and transition frames without triggering state changes.

```python
class MuteController:
    def __init__(self):
        self.state = 'CONTENT'
        self.consecutive_ad = 0
        self.consecutive_content = 0
        self.muted_at_volume = None

    def update(self, p_ad: float):
        vol = get_volume()

        # User override detection
        if self.state == 'AD' and self.muted_at_volume is not None:
            if vol['output_volume'] != self.muted_at_volume:
                self.state = 'CONTENT'
                self.consecutive_ad = 0
                self.consecutive_content = 0
                self.muted_at_volume = None
                return  # user is in control; don't touch audio

        if p_ad >= AD_THRESHOLD:
            self.consecutive_ad += 1
            self.consecutive_content = 0
        elif p_ad < CONTENT_THRESHOLD:
            self.consecutive_content += 1
            self.consecutive_ad = 0
        else:
            # Uncertain zone: decay both counters
            self.consecutive_ad = max(0, self.consecutive_ad - 1)
            self.consecutive_content = max(0, self.consecutive_content - 1)

        if self.state == 'CONTENT' and self.consecutive_ad >= MUTE_DELAY:
            mute()
            self.muted_at_volume = vol['output_volume']
            self.state = 'AD'

        elif self.state == 'AD' and self.consecutive_content >= UNMUTE_DELAY:
            unmute()
            self.muted_at_volume = None
            self.state = 'CONTENT'
```

### Parameter rationale

| Parameter | Default | Reasoning |
|---|---|---|
| `ad_threshold` | 0.60 | Requires moderate confidence before counting as ad |
| `content_threshold` | 0.40 | Requires moderate confidence before counting as content |
| `mute_delay` | 2 | 2 × 5s = 10s of consistent ad frames before muting |
| `unmute_delay` | 3 | 3 × 5s = 15s of consistent content frames before unmuting |

The asymmetry (faster to mute than unmute) is intentional: briefly hearing an ad is less annoying than briefly losing game audio.

---

## 9. Hard Frame Export for Relabeling

Frames where classifier confidence is low (probability in the uncertain zone) are saved to `hard_frames/` and registered in the database for later labeling.

**In `monitor.py`:**

```python
HARD_FRAME_THRESHOLD_LOW = 0.35
HARD_FRAME_THRESHOLD_HIGH = 0.65
MAX_HARD_FRAMES_PER_HOUR = 60  # rate limit

if HARD_FRAME_THRESHOLD_LOW <= p_ad <= HARD_FRAME_THRESHOLD_HIGH:
    if hard_frames_this_hour < MAX_HARD_FRAMES_PER_HOUR:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        dest = os.path.join(hard_frame_dir, f'{ts}.jpg')
        shutil.copy('/tmp/hush_frame.jpg', dest)
        store.add_screenshot(
            filename=f'hard_frames/{ts}.jpg',
            captured_at=datetime.now().isoformat(),
            source='hard_frame'
        )
        hard_frames_this_hour += 1
```

The `label.py` GUI calls `store.get_all()`, which includes `source='hard_frame'` rows. Hard frames appear alongside Phase 1 training frames in the grid and are labeled the same way. On the next `train.py` run, they are included automatically.

---

## 10. Running the System

### Phase 1 workflow

```bash
# Terminal 1: capture frames while watching a stream (Ctrl+C to stop)
python capture.py --interval 23 --output-dir screenshots --db hush.db

# After capturing (aim for 100+ labeled frames minimum):
python label.py --db hush.db --screenshot-dir screenshots

# After labeling:
python train.py --db hush.db --classifier rf --n-folds 5
# Check CV accuracy output; retrain with more data if accuracy < 90%
```

### Phase 2 workflow

```bash
# Run in background while streaming
python monitor.py --model models/classifier.joblib --interval 5 --db hush.db &

# After each streaming session, relabel hard frames and retrain:
python label.py --db hush.db   # hard_frame rows appear automatically
python train.py --db hush.db   # retrain with expanded dataset
# Kill and restart monitor.py to load the new model
```

---

## 11. Potential Issues and Mitigations

**Screen Recording permission (macOS 13+):** `screencapture` requires Screen Recording permission. The first run will prompt automatically. If denied, grant it in System Preferences → Privacy & Security → Screen Recording for Terminal.app or the Python executable.

**DRM and screen capture:** macOS `screencapture` bypasses most browser-based stream DRM (which protects the video bitstream, not the screen framebuffer). Verify this holds for the specific streaming service during Phase 1.

**CPU-only inference (Intel Mac):** This machine is Intel x86, so torch runs CPU-only. ResNet-50 inference takes ~100–200ms per frame — well within the 5-second monitoring interval. No device configuration needed; the code can simply use `device = torch.device('cpu')`. The MPS check (`torch.backends.mps.is_available()`) can be left in as dead code if the project is ever run on Apple Silicon, but on this machine it returns False.

**Class imbalance:** Ads occupy roughly 20–30% of a sports stream. Over 2 hours at 23-second intervals (~313 frames), expect ~65–90 ad frames and ~225–250 content frames — a mild ~3:1 imbalance managed by `class_weight='balanced'`.

**tkinter garbage collection:** All `ImageTk.PhotoImage` objects must be held in a list or instance variable. Python's GC will collect them otherwise, producing blank thumbnails with no error message.

**Monitor restart on model update:** `monitor.py` loads the model once at startup. To use a retrained model, kill and restart the process. A future enhancement could watch the model file's mtime and hot-reload.

**Multiple monitors:** The `-m` flag in `screencapture` captures only the main display. If the stream is on a secondary monitor, remove `-m` (captures all displays into one composite image) or add a `--display` argument that passes `-D <n>` to `screencapture` to target a specific display index.

**Model drift:** A classifier trained on one sport/service will degrade silently on others. If you start watching a different league or switch streaming service, retrain. The hard frame mechanism helps here — uncertain frames accumulate automatically and can be labeled in a batch.
