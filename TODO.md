# Hush — TODO

## Feature extractor upgrade

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
