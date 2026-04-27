"""
PEFTTester — CLIP-in-the-loop evaluation for a trained CompositePEFT model.

Reuses pooling and metric helpers from evaluation.tester.Tester by accessing
the methods unbound and passing self (which carries the same `aggregation`
and `softmax_temp` attributes the helpers expect).

See peft/IMPLEMENTATION_PLAN.md §5 Step 8.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.tester import Tester as _BaseTester  # noqa: E402  (for helpers)


class PEFTTester:
    def __init__(self, config: dict, model, logger):
        self.config = config
        self.logger = logger
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.model.eval()

        self.aggregation = config.get("window_aggregation", "mean")
        if self.aggregation not in ("mean", "max", "softmax"):
            raise ValueError(
                f"window_aggregation must be mean/max/softmax, got {self.aggregation!r}"
            )
        self.softmax_temp = float(config.get("softmax_temp", 1.0))

        amp_dtype = config.get("clip", {}).get("amp_dtype", "fp16")
        self.amp_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[amp_dtype]
        self.amp_enabled = self.amp_dtype != torch.float32

    def _autocast(self):
        return autocast(
            self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled
        )

    @torch.no_grad()
    def _forward_all(self, dataloader, shuffle_frames: bool = False):
        """Forward over the test loader.
        Test loader yields x [B, W, T, 3, H, W], label, video_id, label_cat.
        We flatten windows into the batch dim for forward, then expand labels
        to match for per-window metric computation.
        """
        all_logits, all_labels, all_keys = [], [], []
        total_loss = 0.0

        for x, label, video_id, label_cat in dataloader:
            x = x.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            B, W, T, C, H, Wd = x.shape

            if shuffle_frames:
                # Shared permutation per batch is fine — shuffling the temporal
                # axis is the ablation, the same perm across windows preserves
                # batch-level statistics.
                perm = torch.randperm(T, device=x.device)
                x = x[:, :, perm, :, :, :]

            x_flat = x.reshape(B * W, T, C, H, Wd)

            with self._autocast():
                logits = self.model(x_flat)               # [B*W, num_classes]

            label_flat = label.repeat_interleave(W)       # [B*W]
            loss = F.cross_entropy(logits.float(), label_flat)
            total_loss += float(loss.item())

            all_logits.append(logits.float().cpu())
            all_labels.append(label_flat.cpu())
            for lc, vid in zip(label_cat, video_id):
                all_keys.extend([(lc, vid)] * W)

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        avg_loss = total_loss / max(len(dataloader), 1)
        return logits, labels, all_keys, avg_loss

    def evaluate(self, dataloader, shuffle_frames: bool = False) -> dict:
        tag = "shuffled" if shuffle_frames else "standard"
        self.logger.info(f"Evaluating ({tag})...")

        logits, labels, keys, loss = self._forward_all(dataloader, shuffle_frames)
        probs = torch.softmax(logits, dim=1).numpy()
        labels_np = labels.numpy()

        # Reuse helpers from the base Tester. They reference self.aggregation
        # and self.softmax_temp; both are set on this PEFTTester instance.
        per_window = _BaseTester._compute_metrics(self, probs[:, 1], labels_np, loss=loss)
        v_prob, v_label, _ = _BaseTester._pool_windows_per_video(
            self, probs, labels_np, keys
        )
        per_video = _BaseTester._compute_metrics(self, v_prob, v_label)

        result = {
            "per_window": per_window,
            "per_video": per_video,
            "num_windows": int(labels_np.shape[0]),
            "num_videos": int(v_label.shape[0]),
            "aggregation": self.aggregation,
            "shuffle_frames": bool(shuffle_frames),
        }
        self.logger.info(
            f"  per-window: acc={per_window['accuracy']:.4f}  "
            f"auc={per_window['auc']:.4f}  "
            f"loss={per_window.get('loss', float('nan')):.4f}"
        )
        self.logger.info(
            f"  per-video : acc={per_video['accuracy']:.4f}  "
            f"auc={per_video['auc']:.4f}  "
            f"acc@best={per_video['acc_at_best_thresh']:.4f} "
            f"(thr={per_video['best_thresh']:.3f})"
        )
        return result

    def save_results(
        self,
        out_dir: Path,
        result: dict,
        eval_config_dict: dict,
        shuffled: Optional[dict] = None,
    ) -> None:
        out_dir = Path(out_dir)
        cfg = self.config
        results = {
            "evaluated_utc": datetime.now(timezone.utc).isoformat(),
            "trained_model_dir": cfg.get("trained_model_dir"),
            "split_file": cfg.get("split_file"),
            "rearrange_json": cfg.get("rearrange_json"),
            "dataset_name": cfg.get("dataset_name"),
            "split": cfg.get("split"),
            "num_frames": cfg.get("num_frames"),
            "window_aggregation": self.aggregation,
            "softmax_temp": self.softmax_temp if self.aggregation == "softmax" else None,
            "seed": cfg.get("seed"),
            "standard": result,
            "shuffled": shuffled,
        }
        with open(out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        with open(out_dir / "eval_config.json", "w", encoding="utf-8") as f:
            json.dump(eval_config_dict, f, indent=2)
        self.logger.info(f"Wrote results.json and eval_config.json to {out_dir}")
