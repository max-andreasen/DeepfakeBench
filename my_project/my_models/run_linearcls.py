
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from linear_cls import LinearClassifier
from data_loader import DeepfakeDataset

"""
AI generated code for quick set up of baseline. Not strictly part of study. 
Evaluator for the linear classifier baseline.

Mirrors run_transformer.py — same metrics, same frame-shuffle test,
same results.json format — so runs are directly comparable.
"""


def _infer_model_config(state_dict):
    # the state dict only has classifier.weight and classifier.bias
    clip_embed_dim = int(state_dict["classifier.weight"].shape[1])
    num_classes = int(state_dict["classifier.weight"].shape[0])
    return clip_embed_dim, num_classes


class Evaluator:
    def __init__(self, model_dir, data_split_file, num_frames=32, batch_size=32, num_workers=4):
        self.model_dir = Path(model_dir)
        self.data_split_file = data_split_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        state_dict = torch.load(self.model_dir / "trained_model.pth", map_location=self.device)
        clip_embed_dim, num_classes = _infer_model_config(state_dict)
        print(f"Loaded model: clip_embed_dim={clip_embed_dim}, num_classes={num_classes}")

        self.model = LinearClassifier(
            clip_embed_dim=clip_embed_dim,
            num_classes=num_classes,
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        test_dataset = DeepfakeDataset(data_split_file=data_split_file, split="test", num_frames=num_frames)
        self.loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def evaluate(self, single_frame=False):
        label = "single_frame" if single_frame else "standard"
        print(f"\n--- Evaluating ({label}) ---")

        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        tp = tn = fp = fn = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.loader:
                inputs = inputs.to(self.device)   # [B, n_windows, T, D]
                labels = labels.to(self.device)

                B, n_windows, T, D = inputs.shape

                if single_frame:
                    # pick one random frame per video from the full flattened sequence
                    all_frames = inputs.reshape(B, n_windows * T, D)
                    idx = torch.randint(0, n_windows * T, (B,), device=self.device)
                    picked = all_frames[torch.arange(B, device=self.device), idx]  # [B, D]
                    # reshape to [B, 1, 1, D] so the window loop below still works unchanged
                    inputs = picked.reshape(B, 1, 1, D)
                    B, n_windows, T, D = inputs.shape

                # flatten windows into batch, run model, average logits back per video
                logits_flat = self.model(inputs.reshape(B * n_windows, T, D))   # [B*n_windows, num_classes]
                logits = logits_flat.reshape(B, n_windows, -1).mean(dim=1)      # [B, num_classes]

                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)

                probs = torch.softmax(logits, dim=1)[:, 1]
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())

                total_loss += loss.item() * B
                total_samples += B
                total_correct += int((preds == labels).sum().item())

                tp += int(((preds == 1) & (labels == 1)).sum().item())
                tn += int(((preds == 0) & (labels == 0)).sum().item())
                fp += int(((preds == 1) & (labels == 0)).sum().item())
                fn += int(((preds == 0) & (labels == 1)).sum().item())

        avg_loss = total_loss / total_samples
        acc = total_correct / total_samples
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        auc = roc_auc_score(
            torch.cat(all_labels).numpy(),
            torch.cat(all_probs).numpy(),
        )

        print(f"Test samples : {total_samples}")
        print(f"Loss         : {avg_loss:.4f}")
        print(f"Accuracy     : {acc:.4f}")
        print(f"AUC          : {auc:.4f}")
        print(f"Precision    : {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
        print(f"Confusion    : TP={tp}  TN={tn}  FP={fp}  FN={fn}")

        return {
            "single_frame": single_frame,
            "test_samples": total_samples,
            "loss": round(avg_loss, 6),
            "accuracy": round(acc, 6),
            "auc": round(float(auc), 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        }

    def run(self, single_frame_test=True):
        results = {
            "evaluated_utc": datetime.now(timezone.utc).isoformat(),
            "data_split_file": self.data_split_file,
            "standard": self.evaluate(single_frame=False),
        }
        if single_frame_test:
            # one random frame per video — if this matches standard, per-frame signal dominates
            results["single_frame"] = self.evaluate(single_frame=True)

        results_path = self.model_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    evaluator = Evaluator(
        model_dir="models/trained/linearcls_dim768_constant_lr1e4_epochs50",
        data_split_file="clip/embeddings/celeb_df_ViT-L-14-336-quickgelu_dim768_T96_aligned_0/splits/test20_val10_seed0.csv",
        num_frames=32,
    )
    evaluator.run()
