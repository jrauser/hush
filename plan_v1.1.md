# Hush v1.1 Plan

## Part 1: Upgrade to Python 3.13 + Modern Dependencies

### Goal

Remove version pins that were only necessary for the original Intel Mac environment. The project now runs on Apple Silicon (M5 Pro, arm64), where PyTorch 2.5+ has native wheels and numpy 2.x is fully supported. Upgrading to Python 3.13 with unpinned dependencies gets us current libraries, better performance (MPS improvements in recent torch releases), and eliminates the fragile version ceiling that would rot over time.

---

## Scope

This is a dependency-only change. No application code changes are expected beyond whatever breaks during testing (if anything).

**In scope:**
- New Python 3.13 venv
- Updated `requirements.txt` with relaxed version bounds
- Verify all six scripts work: `capture.py`, `label.py`, `label_store.py`, `train.py`, `monitor.py`, `plot_roc.py`
- Verify trained model still loads and produces equivalent predictions
- Update `README.md` and `implementation_notes.md` to reflect the new environment

**Out of scope:**
- Code refactoring, new features, package reorganization (that's the PyPI TODO item)
- Switching feature extractors (DINOv2, etc.)

---

## Current State

| Package | Current Pin | Installed | Reason for Pin |
|---------|------------|-----------|----------------|
| Python | 3.12 | 3.12.12 | 3.13 lacked torch wheels on Intel Mac |
| torch | `>=2.0,<2.5` | 2.2.2 | 2.5+ dropped Intel Mac wheels |
| torchvision | `>=0.15,<0.20` | 0.17.2 | Pinned to match torch range |
| numpy | `>=1.24,<2` | 1.26.4 | torch <2.5 required numpy 1.x |
| Pillow | `>=10.0` | 12.1.1 | No upper bound (already fine) |
| scikit-learn | `>=1.3` | 1.8.0 | No upper bound (already fine) |
| joblib | `>=1.3` | 1.5.3 | No upper bound (already fine) |
| tqdm | `>=4.0` | 4.67.3 | No upper bound (already fine) |
| flask | `>=3.0` | ? | No upper bound (already fine) |
| matplotlib | *(missing)* | installed | Used by `plot_roc.py`, never added to requirements |

---

## Target State

| Package | New Pin | Notes |
|---------|---------|-------|
| Python | 3.13 | `python-tk@3.13` also needed for tkinter |
| torch | `>=2.5` | Apple Silicon wheels; MPS improvements |
| torchvision | `>=0.20` | Matched to torch >=2.5 |
| numpy | `>=2.0` | torch 2.5+ supports numpy 2.x |
| Pillow | `>=10.0` | Unchanged |
| scikit-learn | `>=1.3` | Unchanged |
| joblib | `>=1.3` | Unchanged |
| tqdm | `>=4.0` | Unchanged |
| flask | `>=3.0` | Unchanged |
| matplotlib | `>=3.8` | Adding the missing dependency |

Lower bounds ensure we don't accidentally resolve ancient versions. No upper bounds — we're on a supported platform now and want rolling upgrades.

---

## Implementation Steps

### 1. Verify prerequisites

Confirm `python3.13` and `python-tk@3.13` are installed via Homebrew. The machine already has `python@3.13` (3.13.12). Check tkinter works:

```bash
python3.13 -c "import tkinter; print(tkinter.TkVersion)"
```

If tkinter is missing: `brew install python-tk@3.13`.

### 2. Create a new venv

```bash
uv venv --python python3.13 .venv-3.13
```

Keep `.venv` intact as a fallback until everything is validated.

### 3. Update `requirements.txt`

```
Pillow>=10.0
torch>=2.5
torchvision>=0.20
scikit-learn>=1.3
joblib>=1.3
numpy>=2.0
tqdm>=4.0
flask>=3.0
matplotlib>=3.8
```

### 4. Install into the new venv

```bash
.venv-3.13/bin/python -m pip install -r requirements.txt
```

Or with uv:
```bash
VIRTUAL_ENV=.venv-3.13 uv pip install -r requirements.txt
```

### 5. Smoke-test each component

Run each check against the new venv:

| Test | Command | What to verify |
|------|---------|---------------|
| Imports | `python -c "import torch, torchvision, numpy, sklearn, flask, PIL, matplotlib, tkinter"` | No import errors |
| MPS | `python -c "import torch; print(torch.backends.mps.is_available())"` | `True` |
| ResNet-50 weights | `python -c "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"` | Loads from cache (same weights file) |
| Feature extraction | `python train.py --db hush.db --n-folds 5` | Completes; compare AUC to current model |
| Model loading | `python -c "import joblib; m = joblib.load('models/classifier.joblib'); print(m.keys())"` | Loads without error |
| Monitor startup | `python monitor.py --model models/classifier.joblib` | GUI opens, classifies a frame |
| Labeler | `python label.py --db hush.db` | GUI opens, thumbnails render |
| Capture | `python capture.py --interval 5 --timeout 0.001` | Takes one screenshot, registers in DB |
| ROC plot | `python plot_roc.py models/classifier_roc.npz` | Renders without error |

### 6. Validate model equivalence

The existing `models/classifier.joblib` was trained under numpy 1.x. scikit-learn models serialized with joblib may not load cleanly across major numpy versions. Test:

```bash
.venv-3.13/bin/python -c "
import joblib, numpy as np
m = joblib.load('models/classifier.joblib')
print(type(m['classifier']))
print('Classes:', m['classifier'].classes_)
"
```

If it fails, retrain under the new environment (the training data in `hush.db` is the source of truth, not the serialized model). The feature cache in `feature_cache/` should also be regenerated — delete it and let `train.py` re-extract.

### 7. Swap venvs

```bash
rm -rf .venv
mv .venv-3.13 .venv
```

### 8. Update documentation

- **`README.md`**: Change `python@3.12` / `python-tk@3.12` references to `@3.13`
- **`implementation_notes.md`**: Add a v1.1 section documenting the upgrade, new package versions, and any issues encountered
- **`TODO.md`**: Mark the Python 3.13 upgrade item as done

---

## Risks and Mitigations

**Model deserialization across numpy versions.** joblib-serialized scikit-learn models embed numpy arrays. A numpy 1.x → 2.x jump can break deserialization. Mitigation: retrain if loading fails. Training data lives in SQLite and is version-independent.

**Feature cache invalidation.** The `.npy` cache files in `feature_cache/` are numpy arrays that may not load across major versions. Mitigation: delete the cache and re-extract. This takes a few minutes on MPS but is fully automated.

**tkinter / tcl-tk version mismatch.** The v1.0 implementation hit this exact problem when a brew upgrade removed tcl-tk 8.6. `python-tk@3.13` links against tcl-tk 9.x. Mitigation: verify `import tkinter` works before proceeding.

**torch API deprecations in 2.5+.** We use a small torch surface area (ResNet-50 inference, MPS device, `nn.Sequential`, `transforms`). These are stable APIs unlikely to break. Mitigation: the smoke tests in step 5 will catch any issues.

---

## Rollback

The old `.venv` (Python 3.12) remains until step 7. If anything fails that can't be resolved quickly, keep the old venv and the original `requirements.txt` (recoverable from git).

---

## Outcome (2026-03-29)

Completed successfully. The Homebrew migration from x86_64 (`/usr/local`) to arm64 (`/opt/homebrew`) was an unplanned prerequisite — the existing `python@3.13` was an x86_64 build carried over from an Intel laptop migration, and PyTorch has no x86_64 macOS wheels for 2.5+.

All smoke tests passed. The existing trained model and feature cache loaded across the numpy 1.x → 2.x boundary without issues. Retraining produced equivalent CV metrics. See `implementation_notes.md` for full details.

---

## Part 2: Unit Tests + Package Layout

### Goal

Add unit tests for the two components with real testable logic (`label_store.py` and the mute state machine), and reorganize the project into a proper Python package layout. The package layout is a prerequisite for the eventual PyPI distribution but is worth doing now because it gives tests a clean import structure and puts us on standard tooling (`pyproject.toml`, `pytest`).

---

### Package Layout

Reorganize from flat files in the project root to a proper package:

```
hush/
├── pyproject.toml              # replaces requirements.txt
├── src/
│   └── hush/
│       ├── __init__.py
│       ├── label_store.py
│       ├── mute_fsm.py         # extracted from monitor.py
│       ├── capture.py
│       ├── label.py
│       ├── train.py
│       ├── monitor.py
│       └── plot_roc.py
├── tests/
│   ├── conftest.py             # shared fixtures (e.g. in-memory LabelStore)
│   ├── test_label_store.py
│   └── test_mute_fsm.py
├── plan_v1.0.md
├── plan_v1.1.md
├── initial_requirements.md
├── implementation_notes.md
├── README.md
├── TODO.md
└── CLAUDE.md
```

**Key decisions:**

- **`src/` layout.** Prevents accidental imports from the working directory. `pytest` discovers packages under `src/` via `pyproject.toml` config. This is the recommended layout for any package that may eventually be installed.
- **`pyproject.toml` replaces `requirements.txt`.** Dependencies move to `[project.dependencies]`. Dev dependencies (pytest) go in `[project.optional-dependencies]`. The `requirements.txt` file is deleted.
- **No entry points yet.** Scripts are still run as `python -m hush.capture`, etc. Entry points (`[project.scripts]`) are deferred to the PyPI distribution TODO, which also requires picking a new package name.
- **Internal imports change.** E.g. `from label_store import LabelStore` → `from hush.label_store import LabelStore`. Every `.py` file that imports `label_store` needs this update (`capture.py`, `label.py`, `train.py`, `monitor.py`).

**`pyproject.toml` sketch:**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "hush"
version = "1.1.0"
requires-python = ">=3.13"
dependencies = [
    "Pillow>=10.0",
    "torch>=2.5",
    "torchvision>=0.20",
    "scikit-learn>=1.3",
    "joblib>=1.3",
    "numpy>=2.0",
    "tqdm>=4.0",
    "flask>=3.0",
    "matplotlib>=3.8",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

---

### Extract `MuteFSM` from `monitor.py`

The mute state machine is currently woven into `MonitorApp` (lines 131–350 of `monitor.py`). It reads `self.args.mute_delay`, `self.args.unmute_delay`, `self._test_mode`, `self._paused`, and calls `_set_muted()` / `_get_muted()` — all of which couple it to the GUI and OS.

Extract into `mute_fsm.py` as a pure-logic class with no dependencies on tkinter, subprocess, or argparse:

```python
class MuteFSM:
    """Streak-based mute state machine.

    Call update() with each new predicted label. Query state via
    properties. The FSM never touches audio — it only decides.
    The caller (MonitorApp) acts on the decision.
    """

    def __init__(self, mute_delay=2, unmute_delay=3):
        self.mute_delay   = mute_delay
        self.unmute_delay = unmute_delay
        self.muted        = False
        self.override     = False
        self.ad_streak    = 0
        self.ok_streak    = 0

    def update(self, pred_label: str, user_unmuted: bool = False) -> str | None:
        """Feed one frame's prediction. Returns 'mute', 'unmute', or None."""
        ...

    def pause(self):
        """Enter paused state. update() becomes a no-op."""
        ...

    def resume(self):
        """Leave paused state."""
        ...
```

**Design principle:** The FSM is a pure function of its inputs. It returns *what to do* ('mute', 'unmute', or None) rather than doing it. `MonitorApp._update_mute_state()` becomes a thin wrapper:

```python
action = self._fsm.update(entry.pred_label, user_unmuted=<check>)
if action == 'mute':
    _set_muted(True)
elif action == 'unmute':
    _set_muted(False)
```

This makes the FSM fully testable without mocking anything.

---

### Unit Tests

**Framework:** `pytest`. No plugins needed. Run with:

```bash
.venv/bin/python -m pytest tests/ -v
```

#### `test_label_store.py`

All tests use an in-memory SQLite database (`:memory:`). No mocks, no filesystem. Shared fixture in `conftest.py`:

```python
@pytest.fixture
def store():
    return LabelStore(":memory:")
```

**Test cases:**

| Test | What it verifies |
|------|-----------------|
| `test_add_and_retrieve` | `add_screenshot` returns rowid; `get_all` returns it; fields match |
| `test_duplicate_filename_rejected` | Second `add_screenshot` with same filename raises `IntegrityError` |
| `test_get_unlabeled_vs_labeled` | After labeling some, `get_unlabeled` and `get_labeled` partition correctly |
| `test_set_label` | `set_label` updates label and sets `labeled_at` |
| `test_set_labels_bulk` | Bulk label N files at once; all updated |
| `test_clear_labels_bulk` | Undo: clears label and labeled_at back to NULL |
| `test_delete_screenshots` | Rows removed from DB; `get_all` no longer returns them |
| `test_get_by_filename_exists` | Returns dict for known file |
| `test_get_by_filename_missing` | Returns None for unknown file |
| `test_record_training_run` | Insert a training run; verify it's queryable and fields round-trip |
| `test_record_training_run_dict_cm` | Pass confusion_matrix as dict; stored as JSON string |
| `test_label_values` | Only 'ad', 'content', 'neither', 'ignore' are used in practice (not enforced by schema, but test the workflow) |

#### `test_mute_fsm.py`

Pure unit tests — no I/O, no mocks.

**Test cases:**

| Test | What it verifies |
|------|-----------------|
| `test_initial_state` | Starts unmuted, no override, streaks at 0 |
| `test_mute_after_delay` | N consecutive 'ad' frames → returns 'mute' on the Nth |
| `test_no_mute_below_delay` | N-1 consecutive 'ad' frames → still None |
| `test_unmute_after_delay` | After muting, M consecutive 'content' frames → returns 'unmute' |
| `test_no_unmute_below_delay` | M-1 content frames → still muted |
| `test_neither_resets_both_streaks` | 'neither' after ad streak resets ad_streak; mute doesn't fire |
| `test_neither_holds_muted_state` | Already muted + 'neither' → stays muted, ok_streak reset |
| `test_ad_resets_ok_streak` | Alternating ad/content never reaches either threshold |
| `test_user_override` | After muting, `user_unmuted=True` → override mode; stops returning 'mute' |
| `test_override_clears_after_content_run` | Override + unmute_delay content frames → override clears, normal behavior resumes |
| `test_pause_suppresses_actions` | While paused, update() always returns None regardless of streaks |
| `test_resume_continues_from_state` | Pause then resume; accumulated streaks preserved (or reset — match current behavior) |
| `test_mute_unmute_cycle` | Full ad-break → content cycle; verify mute then unmute |

---

### Implementation Order

1. **Create `pyproject.toml`; move source files into `src/hush/`; update imports.** Verify everything still runs (`python -m hush.capture`, etc.).
2. **Extract `MuteFSM` into `src/hush/mute_fsm.py`.** Update `monitor.py` to use it. Verify monitor still works.
3. **Write `tests/test_label_store.py`.** Run, fix any issues found.
4. **Write `tests/test_mute_fsm.py`.** Run, fix any issues found.
5. **Delete `requirements.txt`.** Update README and implementation_notes.
6. **Update `.gitignore`** if needed (e.g. `*.egg-info/`).
