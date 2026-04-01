"""train.py — Extract features, cross-validate, and train the hush ad classifier."""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

import joblib
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from hush.label_store import LabelStore

logger = logging.getLogger(__name__)

# Labels included in training (exclude 'ignore')
LABEL_MAP   = {'ad': 0, 'content': 1, 'neither': 2}
LABEL_NAMES = ['ad', 'content', 'neither']

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── feature extractors ────────────────────────────────────────────────────────

def _select_device():
    """Pick the best available device: MPS → CUDA → CPU."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class _ResNet50Extractor:
    name = 'resnet50'

    def __init__(self, device):
        model = tv_models.resnet50(weights='IMAGENET1K_V2')
        self.net = nn.Sequential(*list(model.children())[:-1])
        self.net.eval().to(device)
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __call__(self, paths, on_progress=None):
        """Return np.ndarray of shape [N, 2048].

        If *on_progress* is provided it is called after each batch as
        ``on_progress("extracting", images_done, total_images)``.
        """
        batches = []
        batch_size = 32
        total = len(paths)
        done = 0
        for start in range(0, total, batch_size):
            imgs = [Image.open(p).convert('RGB') for p in paths[start:start + batch_size]]
            batch = torch.stack([self.transform(im) for im in imgs]).to(self.device)
            with torch.no_grad():
                out = self.net(batch).squeeze(-1).squeeze(-1)  # [B, 2048]
            batches.append(out.cpu().numpy())
            done += len(imgs)
            if on_progress:
                on_progress("extracting", done, total)
        return np.vstack(batches)


class _CLIPExtractor:
    name = 'clip'

    def __init__(self, device):
        import clip
        self.model, self.preprocess = clip.load('ViT-B/32', device=device)
        self.model.eval()
        self.device = device

    def __call__(self, paths, on_progress=None):
        """Return np.ndarray of shape [N, 512].

        If *on_progress* is provided it is called after each batch as
        ``on_progress("extracting", images_done, total_images)``.
        """
        batches = []
        batch_size = 32
        total = len(paths)
        done = 0
        for start in range(0, total, batch_size):
            imgs = [Image.open(p).convert('RGB') for p in paths[start:start + batch_size]]
            batch = torch.stack([self.preprocess(im) for im in imgs]).to(self.device)
            with torch.no_grad():
                out = self.model.encode_image(batch).float()  # [B, 512]
            batches.append(out.cpu().numpy())
            done += len(imgs)
            if on_progress:
                on_progress("extracting", done, total)
        return np.vstack(batches)


EXTRACTORS = {
    'resnet50': _ResNet50Extractor,
    'clip': _CLIPExtractor,
}


def build_extractor(name, device):
    if name not in EXTRACTORS:
        raise ValueError(f'Unknown extractor: {name!r}. Choose from: {list(EXTRACTORS)}')
    return EXTRACTORS[name](device)


# ── result dataclass ─────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    """Returned by ``train_model()`` on success."""
    model_path: str
    roc_path: str
    n_samples: int
    label_counts: dict[str, int]
    cv_accuracy_mean: float
    cv_accuracy_std: float
    cv_f1_mean: float
    cv_auc_mean: float
    cv_auc_std: float
    confusion_matrix: list[list[int]]
    fold_results: list[dict] = field(default_factory=list)
    classifier_type: str = ''
    feature_extractor: str = ''


class TrainingError(Exception):
    """Raised when training cannot proceed (not enough data, etc.)."""


# ── core training function ───────────────────────────────────────────────────

def default_model_path() -> str:
    """Generate a timestamped model output path."""
    return 'models/classifier_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.joblib'


def train_model(
    db_path: str,
    model_out: str | None = None,
    extractor_name: str = 'clip',
    classifier_type: str = 'gbm',
    n_folds: int = 5,
    notes: str | None = None,
    on_progress=None,
) -> TrainResult:
    """Run the full training pipeline and return a :class:`TrainResult`.

    Parameters
    ----------
    db_path : str
        Path to the hush SQLite database.
    model_out : str | None
        Where to save the model bundle (``.joblib``).  Defaults to a
        timestamped path under ``models/``.
    extractor_name : str
        Feature extractor name: ``"clip"`` (default) or ``"resnet50"``.
    classifier_type : str
        ``"gbm"`` or ``"rf"``.
    n_folds : int
        Number of cross-validation folds.
    notes : str | None
        Optional free-text notes stored with the training run.
    on_progress : callable | None
        ``on_progress(stage, current, total)`` called during long-running
        steps.  *stage* is ``"extracting"`` or ``"training"``.

    Raises
    ------
    TrainingError
        If the labeled data is insufficient for training.
    """
    if model_out is None:
        model_out = default_model_path()
    os.makedirs(os.path.dirname(model_out) or '.', exist_ok=True)
    store = LabelStore(db_path)

    # ── 1. Load labeled data ─────────────────────────────────────────────
    rows = store.get_labeled()
    rows = [r for r in rows if r['label'] in LABEL_MAP]
    rows = [r for r in rows if os.path.isfile(r['filename'])]

    if not rows:
        raise TrainingError('No usable labeled data. Label some frames first.')

    counts = {name: sum(1 for r in rows if r['label'] == name) for name in LABEL_NAMES}
    present = [name for name in LABEL_NAMES if counts[name] > 0]

    logger.info('Labeled frames: %d total', len(rows))
    for name in LABEL_NAMES:
        logger.info('  %s: %d', name, counts[name])

    if len(present) < 2:
        raise TrainingError('Need at least 2 classes to train a classifier.')

    for name in present:
        if counts[name] < n_folds:
            raise TrainingError(
                f'Only {counts[name]} "{name}" frames — fewer than n_folds={n_folds}. '
                f'Label more frames or lower n_folds.'
            )

    y = np.array([LABEL_MAP[r['label']] for r in rows])

    # ── 2. Feature extraction (incremental, DB-backed) ───────────────────
    all_ids = [r['id'] for r in rows]
    cached = store.get_features(extractor_name, all_ids)
    missing = [r for r in rows if r['id'] not in cached]

    if on_progress:
        on_progress("extracting", 0, len(missing))

    if missing:
        logger.info('Extracting %s features for %d images (%d cached)',
                     extractor_name, len(missing), len(cached))
        device = _select_device()
        logger.info('Device: %s', device)
        extractor = build_extractor(extractor_name, device)
        missing_paths = [r['filename'] for r in missing]
        new_vectors = extractor(missing_paths, on_progress=on_progress)
        store.save_features_bulk(
            extractor_name,
            [(r['id'], new_vectors[i]) for i, r in enumerate(missing)],
        )
        for i, r in enumerate(missing):
            cached[r['id']] = new_vectors[i]
    else:
        logger.info('All %d features loaded from DB (%s).', len(cached), extractor_name)

    # Assemble X in the same order as rows/y
    X = np.vstack([cached[r['id']] for r in rows])
    logger.info('Feature matrix: %s', X.shape)

    # ── 3. Classifier ────────────────────────────────────────────────────
    if classifier_type == 'rf':
        clf      = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                          random_state=42, n_jobs=-1)
        clf_name = 'RandomForest'
    else:
        clf      = HistGradientBoostingClassifier(class_weight='balanced', random_state=42)
        clf_name = 'HistGradientBoosting'

    active_labels = [i for i, name in enumerate(LABEL_NAMES) if counts[name] > 0]

    # ── 4. Cross-validation ──────────────────────────────────────────────
    logger.info('%d-fold stratified CV (%s)', n_folds, clf_name)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    cm_total     = np.zeros((3, 3), dtype=int)
    all_y_true   = []
    all_y_prob   = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        if on_progress:
            on_progress("training", fold, n_folds)

        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[val_idx])
        y_prob = clf.predict_proba(X[val_idx])

        prob3 = np.zeros((len(val_idx), 3))
        for col, cls in enumerate(clf.classes_):
            prob3[:, cls] = y_prob[:, col]

        cm = confusion_matrix(y[val_idx], y_pred, labels=[0, 1, 2])
        cm_total += cm

        acc = (y_pred == y[val_idx]).mean()
        f1  = f1_score(y[val_idx], y_pred, average='macro', zero_division=0,
                       labels=active_labels)
        if len(active_labels) == 2:
            # Binary case: roc_auc_score expects 1D probability of the
            # higher-indexed class.
            auc = roc_auc_score(y[val_idx], prob3[:, active_labels[1]])
        else:
            auc = roc_auc_score(y[val_idx], prob3[:, active_labels],
                                multi_class='ovr', average='macro',
                                labels=active_labels)

        fold_results.append({'fold': fold, 'acc': acc, 'f1': f1, 'auc': auc})
        all_y_true.extend(y[val_idx])
        all_y_prob.extend(prob3)

    accs = [r['acc'] for r in fold_results]
    f1s  = [r['f1']  for r in fold_results]
    aucs = [r['auc'] for r in fold_results]

    # ── 5. Final model ───────────────────────────────────────────────────
    logger.info('Training final %s on all %d samples...', clf_name, len(y))
    clf.fit(X, y)

    bundle = {
        'classifier':          clf,
        'extractor':           extractor_name,
        'label_map':           LABEL_MAP,
        'label_names':         LABEL_NAMES,
        'trained_at':          datetime.now().isoformat(),
        'n_training_samples':  len(y),
    }
    joblib.dump(bundle, model_out)
    logger.info('Model saved → %s', model_out)

    roc_path = model_out.replace('.joblib', '_roc.npz')
    np.savez(roc_path,
             y_true=np.array(all_y_true),
             y_prob=np.array(all_y_prob))
    logger.info('ROC data saved → %s', roc_path)

    # ── 6. Record in DB ──────────────────────────────────────────────────
    store.record_training_run(
        model_path=model_out,
        n_folds=n_folds,
        cv_accuracy_mean=float(np.mean(accs)),
        cv_accuracy_std=float(np.std(accs)),
        cv_f1_mean=float(np.mean(f1s)),
        cv_auc_mean=float(np.mean(aucs)),
        cv_auc_std=float(np.std(aucs)),
        confusion_matrix=cm_total.tolist(),
        n_ad=counts['ad'],
        n_not_ad=counts['content'] + counts['neither'],
        classifier_type=clf_name,
        feature_extractor=extractor_name,
        notes=notes,
    )
    logger.info('Training run recorded in database.')

    return TrainResult(
        model_path=model_out,
        roc_path=roc_path,
        n_samples=len(y),
        label_counts=counts,
        cv_accuracy_mean=float(np.mean(accs)),
        cv_accuracy_std=float(np.std(accs)),
        cv_f1_mean=float(np.mean(f1s)),
        cv_auc_mean=float(np.mean(aucs)),
        cv_auc_std=float(np.std(aucs)),
        confusion_matrix=cm_total.tolist(),
        fold_results=fold_results,
        classifier_type=clf_name,
        feature_extractor=extractor_name,
    )


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train the hush ad classifier.')
    parser.add_argument('--db',         default='hush.db')
    parser.add_argument('--model-out',  default=None,
                        help='Output path (default: models/classifier_YYYYMMDD_HHMMSS.joblib)')
    parser.add_argument('--extractor',  default='resnet50', choices=['resnet50'])
    parser.add_argument('--classifier', default='gbm', choices=['rf', 'gbm'])
    parser.add_argument('--n-folds',    type=int, default=5)
    parser.add_argument('--notes',      default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    try:
        result = train_model(
            db_path=args.db,
            model_out=args.model_out,
            extractor_name=args.extractor,
            classifier_type=args.classifier,
            n_folds=args.n_folds,
            notes=args.notes,
        )
    except TrainingError as exc:
        print(str(exc))
        sys.exit(1)

    # Print summary table
    print(f'\nLabeled frames: {result.n_samples} total')
    for name in LABEL_NAMES:
        print(f'  {name:10s}: {result.label_counts[name]}')

    print(f'\n{args.n_folds}-fold stratified CV ({result.classifier_type})\n')
    print(f'{"Fold":>5} | {"Val Acc":>8} | {"Val F1":>8} | {"Val AUC":>8}')
    print('-' * 42)
    for r in result.fold_results:
        print(f'{r["fold"]:>5} | {r["acc"]:>8.3f} | {r["f1"]:>8.3f} | {r["auc"]:>8.3f}')
    print('-' * 42)
    print(f'{"Mean":>5} | {result.cv_accuracy_mean:>8.3f} | {result.cv_f1_mean:>8.3f} | {result.cv_auc_mean:>8.3f}')
    print(f'{"Std":>5} | {result.cv_accuracy_std:>8.3f} |          | {result.cv_auc_std:>8.3f}')

    cm = result.confusion_matrix
    col_w = 10
    print('\nConfusion matrix (rows=actual, cols=predicted, summed across folds):')
    print(' ' * 12 + ''.join(f'{n:>{col_w}}' for n in LABEL_NAMES))
    for i, name in enumerate(LABEL_NAMES):
        print(f'{name:>12}' + ''.join(f'{cm[i][j]:>{col_w}}' for j in range(3)))

    print(f'\nModel saved → {result.model_path}')
    print(f'ROC data saved → {result.roc_path}')


if __name__ == '__main__':
    main()
