"""plot_roc.py — Plot ad/not-ad ROC curve from saved cross-validation data."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# Label map must match train.py
AD_CLASS = 0  # 'ad' → 0


def main():
    parser = argparse.ArgumentParser(description='Plot ad/not-ad ROC curve.')
    parser.add_argument('--roc-data', default='models/classifier_roc.npz')
    parser.add_argument('--out', default=None,
                        help='Save figure to file instead of displaying (e.g. roc.png)')
    args = parser.parse_args()

    d = np.load(args.roc_data)
    y_true = d['y_true']   # shape [N], values 0/1/2
    y_prob = d['y_prob']   # shape [N, 3]

    # Binary: ad vs everything else
    y_binary = (y_true == AD_CLASS).astype(int)
    y_score  = y_prob[:, AD_CLASS]

    fpr, tpr, thresholds = roc_curve(y_binary, y_score)
    auc = roc_auc_score(y_binary, y_score)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f'Ad vs. rest  (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)

    # Annotate a few threshold values
    for target_thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        idx = np.searchsorted(thresholds[::-1], target_thr)
        idx = len(thresholds) - 1 - idx
        idx = np.clip(idx, 0, len(thresholds) - 1)
        ax.annotate(
            f'{target_thr:.1f}',
            xy=(fpr[idx], tpr[idx]),
            xytext=(fpr[idx] + 0.03, tpr[idx] - 0.04),
            fontsize=8,
            color='steelblue',
            arrowprops=dict(arrowstyle='->', color='steelblue', lw=0.8),
        )
        ax.plot(fpr[idx], tpr[idx], 'o', color='steelblue', markersize=5)

    ax.set_xlabel('False Positive Rate  (content/neither muted)')
    ax.set_ylabel('True Positive Rate  (ads caught)')
    ax.set_title('ROC — ad detection (cross-validation holdout)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Print a threshold table
    print(f'\n{"Threshold":>10} | {"TPR (recall)":>14} | {"FPR":>8} | {"Precision*":>12}')
    print('-' * 52)
    n_pos = y_binary.sum()
    n_neg = len(y_binary) - n_pos
    for target_thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        idx = np.searchsorted(thresholds[::-1], target_thr)
        idx = len(thresholds) - 1 - idx
        idx = np.clip(idx, 0, len(thresholds) - 1)
        tp = tpr[idx] * n_pos
        fp = fpr[idx] * n_neg
        prec = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        print(f'{target_thr:>10.1f} | {tpr[idx]:>14.3f} | {fpr[idx]:>8.3f} | {prec:>12.3f}')
    print('* Precision estimated from cross-validation holdout distribution')

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f'\nSaved → {args.out}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
