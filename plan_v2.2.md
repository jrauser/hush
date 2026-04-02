# Hush v2.2 Plan — Fish for Labeling Errors

**Status:** DRAFT

---

## Overview

When a model is trained, it implicitly scores its own training data: for each
labeled frame, the model assigns a probability to each class. A frame where
`P(true_label)` is low — the model disagrees with the human — is a candidate
for a mislabeling error.

v2.2 surfaces these suspect frames by adding a **score sort** to the Bulk
Capture browser. The user selects "Suspicious first" from a sort dropdown and
sees the frames the model is least confident about at the top of the grid.

---

## Core idea: what "score" means

For each labeled frame:

```
confidence = clf.predict_proba(features)[true_label_index]
```

A frame labeled "ad" where the model gives it `P(ad) = 0.12` is suspicious.
Sorting ascending by confidence puts the worst disagreements first.

Only labeled frames (ad / content / neither) have a meaningful score. Unlabeled
and "ignore" frames are excluded from score-sorted views.

---

## Why this is cheap

Features are already cached in the `features` table after each training run.
Scoring is just `clf.predict_proba(X)` on the stored vectors — a matrix
multiply against a sklearn GBM or RF classifier. For a few thousand frames this
takes well under a second.

Frames labeled *after* the most recent training run won't have cached features
for the current model's extractor. We skip extraction for them and sort them to
the end of the score-sorted view with a note in the status bar:
`"N frames have no score — retrain to include them."`

No modal progress dialog is needed. Scoring is fast enough to happen in the
background and update the grid without the user noticing any lag. If feature
extraction were ever triggered (e.g. a future "score all" button), we'd add a
progress bar then — but not now (YAGNI).

---

## UI changes

### Sort dropdown in Bulk controls bar

Add a `QComboBox` between the search field and the Start/Stop button:

| Option | Behavior |
|--------|----------|
| Newest first | Current default behavior (reverse chronological) |
| Suspicious first | Sorted ascending by `P(true_label)`; unscored frames at end |

The dropdown is **disabled** when no active model is loaded, with a tooltip:
`"Load a trained model to enable score sort"`.

### Probability display on thumbnails

In score sort mode, each thumbnail caption shows **P(ad)** alongside the label,
exactly as live mode does today — e.g. `"ad  12%"` or `"content  81%"`.

P(ad) is the right value to show because it's interpretable regardless of the
true label and reveals the *direction* of disagreement:
- ad-labeled frame with low P(ad) → model thinks it's not an ad, suspicious
- content-labeled frame with high P(ad) → model thinks it's an ad, suspicious

Showing P(true_label) instead would only confirm the sort order without
telling the user *why* the frame is suspicious.

`FrameThumb` already shows probabilities when `prob_ad` is set; the existing
caption logic requires no changes. `score_labeled_frames()` returns the full
probability vector per frame so the caller can extract both P(true_label) for
sorting and P(ad) for display. When sort reverts to "Newest first", `prob_ad`
is set back to `None` and captions return to label-only.

### Status bar

When score sort is active, the status bar appends:
- Normal case: `"Sorted by score (N scored)"`
- With unscored frames: `"Sorted by score (N scored, M unscored — retrain to include)"`

No modal progress dialog.

---

## Implementation

### 1. `train.py` — `score_labeled_frames()`

New public function alongside `train_model()`:

```python
def score_labeled_frames(
    db_path: str,
    bundle: dict,
) -> dict[int, tuple[float, float]]:
    """
    Score all labeled frames using a trained model bundle.

    Returns {screenshot_id: (p_true_label, p_ad)} for frames with cached
    features. Frames with no cached features are omitted.
    p_true_label is used for sort order; p_ad is used for caption display.
    """
```

Steps:
1. Load all labeled rows (ad/content/neither) from `LabelStore`.
2. Look up cached features via `store.get_features(extractor, all_ids)`.
3. Call `bundle['classifier'].predict_proba(X)` on the cached vectors.
4. For each row, extract `p_true_label` via `bundle['label_map'][true_label]`
   and `p_ad` via `bundle['label_map']['ad']`.
5. Return `{screenshot_id: (p_true_label, p_ad)}`.

This function is pure data — no Qt, no threading. Easy to unit-test.

### 2. `label_store.py` — no changes needed

`get_features()` already supports arbitrary id lists and returns only what's
cached. `score_labeled_frames()` uses the existing API.

### 3. `v2app.py` — wire up the sort

#### Sort combo

Add `self._sort_combo = QComboBox()` to the bulk controls bar with options
`["Newest first", "Suspicious first"]`. Connect `currentTextChanged` to
`_apply_bulk_filter`.

#### `_apply_bulk_filter`

Current flow: filter → reverse (newest first) → paginate.

New flow: filter → sort (by date or by score) → paginate.

When "Suspicious first" is selected:
1. Call `score_labeled_frames()` synchronously (it's fast).
2. Attach `(p_true_label, p_ad)` to rows; unscored rows get `(inf, None)`.
3. Sort ascending by `p_true_label`.
4. When building `FrameResult` objects for the grid, set `prob_ad` from the
   score data (or leave `None` for unscored frames).
5. Proceed to paginate as normal.
6. Update status bar with scored/unscored counts.

Load the active model bundle once (cache it on `self`) and reload it when the
active model changes (the existing `_on_settings_saved` path already handles
model reloads — attach score cache invalidation there).

If no model is loaded or `score_labeled_frames()` raises, fall back to newest-
first silently and log a warning. The combo should be disabled before this
point, but defense-in-depth.

#### Sort combo enable/disable

After any model load/unload, update `self._sort_combo.setEnabled(model_loaded)`
and set the tooltip accordingly.

### 4. Tests

| Test | What it checks |
|------|----------------|
| `test_score_labeled_frames_basic` | Given a tiny in-memory DB with features + a mock classifier, returns correct `{id: confidence}` mapping |
| `test_score_labeled_frames_missing_features` | Frames with no cached features are omitted from the result dict |
| `test_score_labeled_frames_label_map` | Each row's confidence indexes the correct class column |
| `test_apply_bulk_filter_sort_newest` | Rows come back reverse-chronological |
| `test_apply_bulk_filter_sort_score` | Rows come back ascending by score; unscored rows at end |

No new widget tests. The sort combo enable/disable logic is exercised by
checking that `score_labeled_frames` returns `{}` when features table is empty
(combo disabled before that point anyway).

---

## Implementation order

1. `score_labeled_frames()` in `train.py` + tests — pure logic, no UI.
2. Sort combo in bulk controls bar (UI only, no scoring yet — combo present but
   "Suspicious first" falls back to newest-first).
3. Wire scoring to `_apply_bulk_filter` + status bar message.
4. Enable/disable combo based on model load state.
5. Manual smoke test: label ~20 frames, train, switch to "Suspicious first",
   verify most-surprising frames appear first.

---

## Non-goals

- **Extracting features for unscored frames on demand.** If the user wants
  all frames scored they can retrain. Feature extraction is slow; triggering
  it implicitly during a sort would be surprising.
- **Scoring live monitor frames.** The live grid isn't paginated and already
  shows model confidence on each thumbnail.
- **A separate score badge distinct from the probability display.** P(ad) in
  the caption is sufficient; a redundant badge would add visual noise.
- **Cross-validation leave-one-out scores.** LOO scoring is more statistically
  honest (avoids overfit confidence) but 5–10× slower and adds complexity. The
  simple `predict_proba` approach is good enough for surfacing gross labeling
  errors.

---

## Open questions

None. Ready for implementation.
