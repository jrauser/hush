"""train.py — Extract features, cross-validate, and train the hush ad classifier."""

import argparse
import json
import os
import sys
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
from tqdm import tqdm

from hush.label_store import LabelStore

# Labels included in training (exclude 'ignore')
LABEL_MAP   = {'ad': 0, 'content': 1, 'neither': 2}
LABEL_NAMES = ['ad', 'content', 'neither']

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── feature extractors ────────────────────────────────────────────────────────

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

    def __call__(self, paths):
        """Return np.ndarray of shape [N, 2048]."""
        batches = []
        batch_size = 32
        for start in tqdm(range(0, len(paths), batch_size), desc='Extracting', unit='batch'):
            imgs = [Image.open(p).convert('RGB') for p in paths[start:start + batch_size]]
            batch = torch.stack([self.transform(im) for im in imgs]).to(self.device)
            with torch.no_grad():
                out = self.net(batch).squeeze(-1).squeeze(-1)  # [B, 2048]
            batches.append(out.cpu().numpy())
        return np.vstack(batches)


def build_extractor(name, device):
    if name == 'resnet50':
        return _ResNet50Extractor(device)
    # Future: elif name == 'dinov2_vits14': return _DinoV2Extractor(device)
    raise ValueError(f'Unknown extractor: {name!r}')


# ── feature cache ─────────────────────────────────────────────────────────────

CACHE_DIR = 'feature_cache'

def _cache_stem(extractor_name, n):
    return os.path.join(CACHE_DIR, f'{extractor_name}_{n}')

def load_cache(extractor_name, n, paths):
    stem = _cache_stem(extractor_name, n)
    fx, fy, fp = stem + '_X.npy', stem + '_y.npy', stem + '_paths.json'
    if not (os.path.isfile(fx) and os.path.isfile(fy) and os.path.isfile(fp)):
        return None, None
    with open(fp) as f:
        cached_paths = json.load(f)
    if cached_paths != paths:
        return None, None
    return np.load(fx), np.load(fy)

def save_cache(extractor_name, n, paths, X, y):
    os.makedirs(CACHE_DIR, exist_ok=True)
    stem = _cache_stem(extractor_name, n)
    np.save(stem + '_X.npy', X)
    np.save(stem + '_y.npy', y)
    with open(stem + '_paths.json', 'w') as f:
        json.dump(paths, f)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train the hush ad classifier.')
    parser.add_argument('--db',         default='hush.db')
    parser.add_argument('--model-out',  default='models/classifier.joblib')
    parser.add_argument('--extractor',  default='resnet50', choices=['resnet50'])
    parser.add_argument('--classifier', default='gbm', choices=['rf', 'gbm'])
    parser.add_argument('--n-folds',    type=int, default=5)
    parser.add_argument('--notes',      default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.model_out) or '.', exist_ok=True)

    # ── 1. Load labeled data ──────────────────────────────────────────────────
    store = LabelStore(args.db)
    rows = store.get_labeled()
    rows = [r for r in rows if r['label'] in LABEL_MAP]
    rows = [r for r in rows if os.path.isfile(r['filename'])]

    if not rows:
        print('No usable labeled data. Label some frames first.')
        sys.exit(1)

    counts = {name: sum(1 for r in rows if r['label'] == name) for name in LABEL_NAMES}
    present = [name for name in LABEL_NAMES if counts[name] > 0]

    print(f'\nLabeled frames: {len(rows)} total')
    for name in LABEL_NAMES:
        print(f'  {name:10s}: {counts[name]}')

    if len(present) < 2:
        print('\nNeed at least 2 classes to train a classifier.')
        sys.exit(1)

    for name in present:
        if counts[name] < args.n_folds:
            print(f'\nOnly {counts[name]} "{name}" frames — fewer than --n-folds {args.n_folds}.')
            print('Label more frames or lower --n-folds.')
            sys.exit(1)

    paths = [r['filename'] for r in rows]
    y     = np.array([LABEL_MAP[r['label']] for r in rows])

    # ── 2. Feature extraction (with cache) ────────────────────────────────────
    X, y_cached = load_cache(args.extractor, len(rows), paths)
    if X is not None:
        print(f'\nLoaded features from cache ({args.extractor}, {len(rows)} samples).')
    else:
        print(f'\nExtracting {args.extractor} features for {len(rows)} images...')
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        print(f'Device: {device}')
        extractor = build_extractor(args.extractor, device)
        X = extractor(paths)
        save_cache(args.extractor, len(rows), paths, X, y)
        print(f'Feature matrix: {X.shape}')

    # ── 3. Classifier ─────────────────────────────────────────────────────────
    if args.classifier == 'rf':
        clf      = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                          random_state=42, n_jobs=-1)
        clf_name = 'RandomForest'
    else:
        clf      = HistGradientBoostingClassifier(class_weight='balanced', random_state=42)
        clf_name = 'HistGradientBoosting'

    # Use only classes that are present in the data (for roc_auc_score)
    active_labels = [i for i, name in enumerate(LABEL_NAMES) if counts[name] > 0]

    # ── 4. Cross-validation ───────────────────────────────────────────────────
    print(f'\n{args.n_folds}-fold stratified CV ({clf_name})\n')
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    fold_results = []
    cm_total     = np.zeros((3, 3), dtype=int)
    all_y_true   = []
    all_y_prob   = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[val_idx])
        y_prob = clf.predict_proba(X[val_idx])  # [n_val, n_classes]

        # predict_proba columns match clf.classes_, not necessarily [0,1,2]
        # build a full [n_val, 3] prob matrix
        prob3 = np.zeros((len(val_idx), 3))
        for col, cls in enumerate(clf.classes_):
            prob3[:, cls] = y_prob[:, col]

        cm = confusion_matrix(y[val_idx], y_pred, labels=[0, 1, 2])
        cm_total += cm

        acc = (y_pred == y[val_idx]).mean()
        f1  = f1_score(y[val_idx], y_pred, average='macro', zero_division=0,
                       labels=active_labels)
        auc = roc_auc_score(y[val_idx], prob3[:, active_labels],
                            multi_class='ovr', average='macro',
                            labels=active_labels)

        fold_results.append({'fold': fold, 'acc': acc, 'f1': f1, 'auc': auc})
        all_y_true.extend(y[val_idx])
        all_y_prob.extend(prob3)

    accs = [r['acc'] for r in fold_results]
    f1s  = [r['f1']  for r in fold_results]
    aucs = [r['auc'] for r in fold_results]

    print(f'{"Fold":>5} | {"Val Acc":>8} | {"Val F1":>8} | {"Val AUC":>8}')
    print('-' * 42)
    for r in fold_results:
        print(f'{r["fold"]:>5} | {r["acc"]:>8.3f} | {r["f1"]:>8.3f} | {r["auc"]:>8.3f}')
    print('-' * 42)
    print(f'{"Mean":>5} | {np.mean(accs):>8.3f} | {np.mean(f1s):>8.3f} | {np.mean(aucs):>8.3f}')
    print(f'{"Std":>5} | {np.std(accs):>8.3f} | {np.std(f1s):>8.3f} | {np.std(aucs):>8.3f}')

    col_w = 10
    print('\nConfusion matrix (rows=actual, cols=predicted, summed across folds):')
    print(' ' * 12 + ''.join(f'{n:>{col_w}}' for n in LABEL_NAMES))
    for i, name in enumerate(LABEL_NAMES):
        print(f'{name:>12}' + ''.join(f'{cm_total[i, j]:>{col_w}}' for j in range(3)))

    # ── 5. Final model ────────────────────────────────────────────────────────
    print(f'\nTraining final {clf_name} on all {len(y)} samples...')
    clf.fit(X, y)

    bundle = {
        'classifier':          clf,
        'extractor':           args.extractor,
        'label_map':           LABEL_MAP,
        'label_names':         LABEL_NAMES,
        'trained_at':          datetime.now().isoformat(),
        'n_training_samples':  len(y),
    }
    joblib.dump(bundle, args.model_out)
    print(f'Model saved → {args.model_out}')

    roc_path = args.model_out.replace('.joblib', '_roc.npz')
    np.savez(roc_path,
             y_true=np.array(all_y_true),
             y_prob=np.array(all_y_prob))
    print(f'ROC data saved → {roc_path}')

    # ── 6. Record in DB ───────────────────────────────────────────────────────
    store.record_training_run(
        model_path=args.model_out,
        n_folds=args.n_folds,
        cv_accuracy_mean=float(np.mean(accs)),
        cv_accuracy_std=float(np.std(accs)),
        cv_f1_mean=float(np.mean(f1s)),
        cv_auc_mean=float(np.mean(aucs)),
        cv_auc_std=float(np.std(aucs)),
        confusion_matrix=cm_total.tolist(),
        n_ad=counts['ad'],
        n_not_ad=counts['content'] + counts['neither'],
        classifier_type=clf_name,
        feature_extractor=args.extractor,
        notes=args.notes,
    )
    print('Training run recorded in database.')


if __name__ == '__main__':
    main()
