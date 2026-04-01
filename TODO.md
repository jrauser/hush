# Hush — TODO

## README: getting started flow

The current README covers running the app but not the new-user journey.
Add a "Getting Started" section covering the full sequence:
clone → install → load Chrome extension → capture frames → label → train → monitor.

Also note explicitly that the first training run downloads ~350MB of CLIP model
weights.

## Setup script

A `setup.sh` that creates the venv, installs deps, and pre-downloads the CLIP
weights would avoid a surprise 350MB download mid-training.  Single command for
new users, nothing required after that.

## Starter model

Consider committing a pre-trained `models/starter.joblib` trained on a clean
dataset (tab-capture only, no full-screen frames).  Lets new users run the
monitor immediately to see if it works on their stream before committing to the
labeling effort.  Document clearly that it's tuned to specific content and may
need retraining.

## ~~Feature extractor upgrade~~ DONE (2026-03-31)

Try CLIP (ViT-B/32) as a drop-in replacement for ResNet-50 in `build_extractor()`
([src/hush/frame_pipeline.py](src/hush/frame_pipeline.py)).  CLIP's semantic
features may be more discriminative for ad vs. sports content because its training
corpus links visual grammar to text descriptions.  DINOv2 (ViT-S/14) is a fallback
if CLIP doesn't improve accuracy.

**Infrastructure already in place:**
- `features` table has an `extractor` column — no schema changes needed
- `training_runs` table already records the extractor — no schema changes needed

**Work required:**
1. `pip install openai-clip` — add to pyproject.toml
2. Add `"clip"` extractor branch in `build_extractor()` (and update `build_extractor()` in train.py if it has its own copy)
3. Training screen: add extractor picker (resnet50 / clip) so the user can choose before training
4. Model browser: surface the extractor name alongside accuracy metrics so it's visible which extractor a model was trained with
5. Re-extract features from existing labeled data with new extractor, retrain, compare cross-val accuracy vs. ResNet-50 baseline
