
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from data_loader import DeepfakeDataset
from transfomer import Transformer

"""
Evaluator class for running inference on the test set.

Loads the final trained model weights from a models/trained/ run directory,
runs multi-segment inference on the test split (averages logits across windows),
and prints accuracy, precision, recall, F1, and the confusion matrix.

Frame-shuffle test:
    evaluate(shuffle_frames=True) randomly permutes the T frame dimension before
    inference. A model that captures temporal patterns should perform worse when
    frames are shuffled. run() calls both variants and saves combined results.

Label convention:
- 0 => fake
- 1 => real
"""


def _infer_model_config(state_dict):
    clip_embed_dim = int(state_dict["cls_token"].shape[-1])
    num_frames = int(state_dict["positional_encoding"].shape[1] - 1)
    classifier_weights = [
        (k, v) for k, v in state_dict.items()
        if k.startswith("classifier.") and k.endswith(".weight")
    ]
    last_key, last_weight = max(classifier_weights, key=lambda kv: int(kv[0].split(".")[1]))
    num_classes = int(last_weight.shape[0])
    return clip_embed_dim, num_frames, num_classes


class Evaluator:
    def __init__(self, model_dir, data_split_file, batch_size=32, num_workers=4):
        self.model_dir = Path(model_dir)
        self.data_split_file = data_split_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        model_path = self.model_dir / "trained_model.pth"
        state_dict = torch.load(model_path, map_location=self.device)
        clip_embed_dim, num_frames, num_classes = _infer_model_config(state_dict)
        print(f"Loaded model: clip_embed_dim={clip_embed_dim}, num_frames={num_frames}, num_classes={num_classes}")

        self.model = Transformer(
            clip_embed_dim=clip_embed_dim,
            num_frames=num_frames,
            num_classes=num_classes,
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        test_dataset = DeepfakeDataset(data_split_file=data_split_file, split="test", num_frames=num_frames)
        self.loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def evaluate(self, shuffle_frames=False):
        label = "shuffled" if shuffle_frames else "standard"
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
                inputs = inputs.to(self.device)   # [B, n_windows, num_frames, D]
                labels = labels.to(self.device)

                B, n_windows, T, D = inputs.shape

                if shuffle_frames:
                    # permute the T dimension to break temporal order
                    perm = torch.randperm(T, device=self.device)
                    inputs = inputs[:, :, perm, :]

                # flatten windows into batch, run model, average logits back per video
                logits_flat = self.model(inputs.reshape(B * n_windows, T, D))   # [B*n_windows, num_classes]
                logits = logits_flat.reshape(B, n_windows, -1).mean(dim=1)      # [B, num_classes]

                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)

                # probability of class 1 (real) for AUC
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
            "shuffle_frames": shuffle_frames,
            "test_samples": total_samples,
            "loss": round(avg_loss, 6),
            "accuracy": round(acc, 6),
            "auc": round(float(auc), 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        }

    def run(self, frame_shuffle_test=True):
        results = {
            "evaluated_utc": datetime.now(timezone.utc).isoformat(),
            "data_split_file": self.data_split_file,
            "standard": self.evaluate(shuffle_frames=False),
        }
        if frame_shuffle_test:
            results["shuffled"] = self.evaluate(shuffle_frames=True)

        results_path = self.model_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    evaluator = Evaluator(
        model_dir="models/trained/transformer_dim768_cosine_warmup_lr1e4_epochs50_2",
        data_split_file="clip/embeddings/celeb_df_ViT-L-14-336-quickgelu_dim768_T96_0/splits/test20_val10_seed0.csv",
    )
    evaluator.run(frame_shuffle_test=True)
