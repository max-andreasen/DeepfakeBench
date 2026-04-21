"""
Tester class, runs forward passes and computes metrics.

Fully controls model, device, config, logger.

Test loader (DeepfakeTestDataset) loads one video per item:
    x [n_windows, T, D], label, video_id
So a batch is x [B, n_windows, T, D].

The Tester flattens windows into the batch dim before forward, computes per-window metrics,
then aggregatessoftmax probs back per video for per-video metrics.
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

# Entry points differ on what's on sys.path: layer_probe.py / parameter_search.py
# add training/ to the path (so 'utils...' resolves), while evaluation/test.py adds
# the repo root (so 'training.utils...' resolves). Accept either.
try:
    from utils.temporal_transforms import apply_temporal_transform
except ImportError:
    from training.utils.temporal_transforms import apply_temporal_transform

class Tester:
    def __init__(self, config, model, logger):
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.aggregation = config.get('window_aggregation', 'mean')
        if self.aggregation not in ('mean', 'max', 'softmax'):
            raise ValueError(f"window_aggregation must be 'mean', 'max', or 'softmax', got '{self.aggregation}'")
        # Temperature for softmax-weighted pooling. T -> 0 approaches max, T -> inf approaches mean.
        self.softmax_temp = float(config.get('softmax_temp', 1.0))

    @torch.no_grad()
    def _forward_all(self, dataloader, shuffle_frames):
        """Forward passes over the entire dataloader.
        Returns (logits, labels, video_keys, total_loss).
        video_keys is a list of (label_cat, video_id) tuples, one per window.
        Optionally shuffles the temporal axis independently per window.
        """

        all_logits = []
        all_labels = []
        all_video_keys = []
        total_loss = 0.0

        # Discover transform from the dataset — single source of truth with
        # DeepfakeTestDataset. Defaults to 'none' if the dataset predates the field.
        input_transform = getattr(dataloader.dataset, "input_transform", "none")

        for x, label, video_id, label_cat in dataloader:
            x = x.to(self.device)
            label = label.to(self.device)
            B, W, T, D = x.shape

            if shuffle_frames:
                x = x[:, :, torch.randperm(T, device=x.device), :]

            # Apply temporal transform AFTER shuffle so 'diff' on a shuffled
            # window produces noise diffs (the intended behavior for the
            # shuffled-baseline temporal-gap measurement).
            x = apply_temporal_transform(x, input_transform)

            x_flat = x.reshape(-1, x.shape[-2], x.shape[-1])
            logits = self.model(x_flat)                       # [B*W, C]
            label_flat = label.repeat_interleave(W)           # [B*W]
            loss = F.cross_entropy(logits, label_flat)
            total_loss += loss.item()

            # sklearn uses CPU, also off-loads the GPU a bit.
            all_logits.append(logits.cpu())
            all_labels.append(label_flat.cpu())
            # Replicate (label_cat, video_id) W times, preserving per-batch order.
            for lc, vid in zip(label_cat, video_id):
                all_video_keys.extend([(lc, vid)] * W)

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        return logits, labels, all_video_keys, total_loss / max(len(dataloader), 1)


    def _pool_windows_per_video(self, window_probs, window_labels, window_video_keys):
        """Collapse window-level fake-class probs into one prob per video using
        self.aggregation ('mean', 'max', or 'softmax').
        window_video_keys is a list of (label_cat, video_id) tuples; FF++ fakes
        reuse video_id strings across manipulations, so (label_cat, video_id)
        is the real unique key.
        Returns three arrays of length num_videos:
            per_video_probs, per_video_labels, video_keys (in first-seen order).
        """
        # window_probs: [num_windows, num_classes] numpy
        prob_pos = window_probs[:, 1]
        groups = defaultdict(list)          # (label_cat, video_id) -> list of prob_pos
        labels_per_video = {}               # (label_cat, video_id) -> int label
        order = []                          # preserve first-seen order of video keys
        for i, key in enumerate(window_video_keys):
            if key not in labels_per_video:
                labels_per_video[key] = int(window_labels[i])
                order.append(key)
            else:
                # Sanity: same video should always have the same label.
                assert labels_per_video[key] == int(window_labels[i]), \
                    f"Inconsistent labels for video {key}"
            groups[key].append(prob_pos[i])

        if self.aggregation == 'mean':
            agg_fn = np.mean
        elif self.aggregation == 'max':
            agg_fn = np.max
        else:  # 'softmax': weights = softmax(probs / T), score = sum(w * probs)
            T = self.softmax_temp
            def agg_fn(p):
                p = np.asarray(p, dtype=np.float64)
                z = p / T
                z = z - z.max()                 # numerical stability
                w = np.exp(z)
                w /= w.sum()
                return float((w * p).sum())
        per_video_probs = np.array([agg_fn(groups[v]) for v in order], dtype=np.float64)
        per_video_labels = np.array([labels_per_video[v] for v in order], dtype=np.int64)
        return per_video_probs, per_video_labels, order


    def _compute_metrics(self, prob_pos, labels, loss=None):
        """Metrics from a 1-D positive-class prob array + integer labels.
        prob_pos and labels are numpy arrays of equal length."""
        preds = (prob_pos >= 0.5).astype(np.int64)
        acc = float((preds == labels).mean())

        try:
            auc = float(roc_auc_score(labels, prob_pos))
        except ValueError:
            auc = float('nan')

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', pos_label=1, zero_division=0
        )
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        # Best threshold by Youden's J (tpr - fpr); useful under class imbalance.
        try:
            fpr, tpr, thresholds = roc_curve(labels, prob_pos)
            best_idx = (tpr - fpr).argmax()
            best_thresh = float(thresholds[best_idx])
            acc_at_best = float(((prob_pos >= best_thresh) == labels).mean())
        except ValueError:
            best_thresh = float('nan')
            acc_at_best = float('nan')

        out = {
            'accuracy': acc,
            'auc': auc,
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
            'best_thresh': best_thresh,
            'acc_at_best_thresh': acc_at_best,
            'num_samples': int(labels.shape[0]),
        }
        if loss is not None:
            out['loss'] = float(loss)
        return out


    def evaluate(self, dataloader, shuffle_frames=False):
        """Run a full eval pass; return per-window + per-video metric dicts."""
        tag = 'shuffled' if shuffle_frames else 'standard'
        self.logger.info(f"Evaluating ({tag})...")

        logits, labels, video_keys, loss = self._forward_all(dataloader, shuffle_frames)
        probs = torch.softmax(logits, dim=1).numpy()
        labels_np = labels.numpy()

        per_window = self._compute_metrics(probs[:, 1], labels_np, loss=loss)
        v_prob, v_label, _ = self._pool_windows_per_video(probs, labels_np, video_keys)
        per_video = self._compute_metrics(v_prob, v_label)

        input_transform = getattr(dataloader.dataset, "input_transform", "none")
        result = {
            'per_window': per_window,
            'per_video': per_video,
            'num_windows': int(labels_np.shape[0]),
            'num_videos': int(v_label.shape[0]),
            'aggregation': self.aggregation,
            'shuffle_frames': bool(shuffle_frames),
            'input_transform': input_transform,
        }

        self.logger.info(
            f"  per-window: acc={per_window['accuracy']:.4f}  auc={per_window['auc']:.4f}  "
            f"loss={per_window.get('loss', float('nan')):.4f}"
        )
        self.logger.info(
            f"  per-video : acc={per_video['accuracy']:.4f}  auc={per_video['auc']:.4f}  "
            f"acc@best={per_video['acc_at_best_thresh']:.4f} (thr={per_video['best_thresh']:.3f})"
        )
        return result


    def save_results(self, out_dir, standard, shuffled, eval_config_dict):
        """Write results.json + eval_config.json into out_dir.
        test.log is handled by the logger passed in at __init__."""
        out_dir = Path(out_dir)
        cfg = self.config
        results = {
            'evaluated_utc': datetime.now(timezone.utc).isoformat(),
            'trained_model_dir': cfg.get('trained_model_dir'),
            'split_file': cfg.get('split_file'),
            'catalogue_file': cfg.get('catalogue_file'),
            'test_dataset': cfg.get('test_dataset'),
            'num_frames': cfg.get('num_frames'),
            'window_aggregation': self.aggregation,
            'softmax_temp': self.softmax_temp if self.aggregation == 'softmax' else None,
            'seed': cfg.get('seed'),
            'standard': standard,
            'shuffled': shuffled,
        }
        with open(out_dir / 'results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        with open(out_dir / 'eval_config.json', 'w', encoding='utf-8') as f:
            json.dump(eval_config_dict, f, indent=2)
        self.logger.info(f"Wrote results.json and eval_config.json to {out_dir}")
