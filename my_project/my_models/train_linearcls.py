
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, StepLR
from torch.utils.data import DataLoader

from linear_cls import LinearClassifier
from data_loader import DeepfakeDataset

"""
AI generated code for quick set up of baseline. Not strictly part of study. 
Trainer for the linear classifier baseline.

Mirrors train_transformer.py closely — same options for LR scheduler,
weighted loss, etc. — so results are directly comparable.
"""


class Trainer:
    def __init__(
        self,
        data_split_file,
        clip_embed_dim=768,
        num_classes=2,
        num_frames=32,
        lr=1e-4,
        weight_decay=0.0,
        batch_size=32,
        num_epochs=50,
        lr_scheduler="constant",    # "constant" | "cosine_warmup" | "step"
        warmup_epochs=5,            # cosine_warmup: linear warmup length
        step_size=10,               # step: decay every N epochs
        step_gamma=0.5,             # step: multiply LR by this factor
        weighted_loss=False,        # weight CE loss by inverse class frequency
        num_workers=4,
    ):
        self.num_epochs = num_epochs

        self.config = {
            "clip_embed_dim": clip_embed_dim,
            "num_classes": num_classes,
            "num_frames": num_frames,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "lr_scheduler": lr_scheduler,
            "warmup_epochs": warmup_epochs,
            "step_size": step_size,
            "step_gamma": step_gamma,
            "weighted_loss": weighted_loss,
        }

        # build the run directory name, auto-increment if it already exists
        lr_str = f"{lr:.0e}".replace("e-0", "e").replace("e+0", "e")
        run_name = f"linearcls_dim{clip_embed_dim}_{lr_scheduler}_lr{lr_str}_epochs{num_epochs}"
        self.run_dir = Path("models/trained") / run_name
        inc = 1
        while self.run_dir.exists():
            self.run_dir = Path("models/trained") / f"{run_name}_{inc}"
            inc += 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Started training with device: {self.device}")

        self.model = LinearClassifier(
            clip_embed_dim=clip_embed_dim,
            num_classes=num_classes,
        ).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # learning rate scheduler
        if lr_scheduler == "cosine_warmup":
            warmup = LinearLR(self.optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
            cosine = CosineAnnealingLR(self.optimizer, T_max=num_epochs - warmup_epochs, eta_min=0)
            self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        elif lr_scheduler == "step":
            self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=step_gamma)
        else:
            self.scheduler = None   # constant LR

        train_dataset = DeepfakeDataset(data_split_file=data_split_file, split="train", num_frames=num_frames)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        if weighted_loss:
            labels = torch.tensor(train_dataset.labels)
            counts = torch.bincount(labels)
            class_weights = 1.0 / counts.float()
            class_weights = class_weights / class_weights.sum()
            self.loss_func = torch.nn.CrossEntropyLoss(weight=class_weights.to(self.device))
            print(f"Using weighted CE loss: fake={class_weights[0]:.4f}, real={class_weights[1]:.4f}")
        else:
            self.loss_func = torch.nn.CrossEntropyLoss()
            print("Using standard CE loss")

    def _train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for i, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # inputs: [B, T, D] — a single training window per video

            outputs = self.model(inputs)

            loss = self.loss_func(outputs, labels)
            predictions = outputs.argmax(dim=1)
            epoch_correct += (predictions == labels).sum().item()
            epoch_total += labels.size(0)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            running_loss += loss.item()
            epoch_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}], Loss: {running_loss/10:.4f}")
                running_loss = 0.0

        avg_loss = epoch_loss / len(self.train_loader)
        train_acc = epoch_correct / epoch_total
        print(f"Epoch [{epoch+1}/{self.num_epochs}] Summary - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")

    def _store_model(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), self.run_dir / "trained_model.pth")

        run_config = {**self.config, "created_utc": datetime.now(timezone.utc).isoformat()}
        with open(self.run_dir / "run_config.json", "w") as f:
            json.dump(run_config, f, indent=2)

        print(f"Model stored in {self.run_dir}")

    def train(self):
        for epoch in range(self.num_epochs):
            self._train_epoch(epoch)
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"Epoch [{epoch+1}/{self.num_epochs}] LR: {current_lr:.2e}")
        self._store_model()
        print("Training Complete!")


if __name__ == "__main__":
    trainer = Trainer(
        data_split_file="clip/embeddings/celeb_df_ViT-L-14-336-quickgelu_dim768_T96_aligned_0/splits/test20_val10_seed0.csv",
        clip_embed_dim=768,
        num_frames=32,
        num_classes=2,
        lr=1e-4,
        weight_decay=0.0,
        batch_size=32,
        num_epochs=50,
        lr_scheduler="constant",    # "constant" | "cosine_warmup" | "step"
        # warmup_epochs=5,          # used by cosine_warmup
        # step_size=10,             # used by step
        # step_gamma=0.5,           # used by step
        weighted_loss=False,
    )
    trainer.train()
