# Hush v1.1: Upgrade to Python 3.13 + Modern Dependencies

## Goal

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
