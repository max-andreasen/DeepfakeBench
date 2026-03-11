
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, StepLR
from torch.utils.data import DataLoader
from transfomer import Transformer
from data_loader import DeepfakeDataset

"""
Trainer class for the temporal deepfake Transformer.

How the training loop works:
    Runs the forward pass of the Transformer model.
    Calculates the loss using a loss-function.
    Runs the backwards pass (back-prop) to calculate the gradients.
    Applies optimiser, using the gradients to shift the weights.

The training loader:
    This module is used to pre-batch the data, creating a more efficient training
    loop by not having the GPU idle. It also handles the data, e.g. shuffling

Using cross-entropy loss:
    What other options are there?
    What is generally used here?

Using the Adam / AdamW optimiser:
    Updates the weights with the accumulated gradients.
    AdamW is an improved version of Adam, but for this experiment I will try both.
    Also contains hyper-params to be tuned later on.

Hyper-params:
    - Epochs (defaulting to 10).
    - Batch-size.
    - Adam / optimizer.

Do I need accumalation steps?
"""


class Trainer:
    def __init__(
        self,
        data_split_file,
        clip_embed_dim=768,
        num_frames=32,
        num_classes=2,
        num_layers=2,
        lr=1e-4,
        weight_decay=0.0,
        batch_size=32,
        num_epochs=50,
        attn_dropout=0.1,           # dropout inside transformer attention/feedforward sublayers
        mlp_dropout=0.4,            # starting dropout for MLP head (linearly decreases to 0)
        lr_scheduler="constant",    # "constant" | "cosine_warmup" | "step"
        warmup_epochs=5,            # cosine_warmup: linear warmup length
        step_size=10,               # step: decay every N epochs
        step_gamma=0.5,             # step: multiply LR by this factor
        weighted_loss=False,        # weight CE loss by inverse class frequency
        create_checkpoints=False,
        num_workers=4,
    ):
        self.num_epochs = num_epochs
        self.create_checkpoints = create_checkpoints

        # config used for storing the model / checkpoints
        self.config = {
            "clip_embed_dim": clip_embed_dim,
            "num_frames": num_frames,
            "num_classes": num_classes,
            "num_layers": num_layers,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "attn_dropout": attn_dropout,
            "mlp_dropout": mlp_dropout,
            "lr_scheduler": lr_scheduler,
            "warmup_epochs": warmup_epochs,
            "step_size": step_size,
            "step_gamma": step_gamma,
            "weighted_loss": weighted_loss,
        }

        self.lr_type = lr_scheduler

        # construct the run directory name from hyperparams
        # format: transformer_dim768_[LR-TYPE]_lr1e4_epochs50
        lr_str = f"{lr:.0e}"    # e.g. 0.0001 -> "1e-04", cleaned up below
        lr_str = lr_str.replace("e-0", "e").replace("e+0", "e")
        run_name = f"transformer_dim{clip_embed_dim}_{self.lr_type}_lr{lr_str}_epochs{num_epochs}"

        # auto-increment run dir if one with the same name already exists
        base = Path("models/trained") / run_name
        self.run_dir = base
        inc = 1
        while self.run_dir.exists():
            self.run_dir = Path("models/trained") / f"{run_name}_{inc}"
            inc += 1

        self.checkpoint_dir = self.run_dir / "checkpoints"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Started training with device: {self.device}")

        self.model = Transformer(
            clip_embed_dim=clip_embed_dim,
            num_frames=num_frames,
            num_classes=num_classes,
            num_layers=num_layers,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
        ).to(self.device)

        # AdamW is generally better than Adam for Transformers
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # learning rate scheduler
        if lr_scheduler == "cosine_warmup":
            # linear warmup from lr*1e-3 to lr, then cosine decay to 0
            warmup = LinearLR(self.optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
            cosine = CosineAnnealingLR(self.optimizer, T_max=num_epochs - warmup_epochs, eta_min=0)
            self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        elif lr_scheduler == "step":
            # multiply LR by step_gamma every step_size epochs
            self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=step_gamma)
        else:
            self.scheduler = None   # constant LR

        # loads the data into memory with the data loader
        train_dataset = DeepfakeDataset(data_split_file=data_split_file, split="train", num_frames=num_frames)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,       # Shuffle every epoch
            num_workers=num_workers,
        )

        if weighted_loss:
            # weight CE loss by inverse class frequency to counteract class imbalance (fake >> real)
            labels = torch.tensor(train_dataset.labels)
            counts = torch.bincount(labels)         # [n_fake, n_real]
            class_weights = 1.0 / counts.float()
            class_weights = class_weights / class_weights.sum()
            self.loss_func = torch.nn.CrossEntropyLoss(weight=class_weights.to(self.device))
            print(f"Using weighted CE loss: fake={class_weights[0]:.4f}, real={class_weights[1]:.4f}")
        else:
            self.loss_func = torch.nn.CrossEntropyLoss()
            print("Using standard CE loss")



    def _train_epoch(self, epoch):
        self.model.train()      # sets the PyTorch model into training mode
        running_loss = 0.0
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for i, (inputs, labels) in enumerate(self.train_loader):

            # loads the data to GPU
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            outputs = self.model(inputs)        # runs forward pass automatically

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



    def _save_checkpoint(self, epoch):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / f"transformer_epoch_{epoch+1}.pth"
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")



    def _store_model(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)

        model_path = self.run_dir / "trained_model.pth"
        torch.save(self.model.state_dict(), model_path)

        config_path = self.run_dir / "run_config.json"
        run_config = {**self.config, "created_utc": datetime.now(timezone.utc).isoformat()}
        with open(config_path, "w") as f:
            json.dump(run_config, f, indent=2)

        print(f"Model stored in {self.run_dir}")



    def train(self):
        for epoch in range(self.num_epochs):
            self._train_epoch(epoch)
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"] # stored internally with Adam. 
                print(f"Epoch [{epoch+1}/{self.num_epochs}] LR: {current_lr:.2e}")
            if self.create_checkpoints:
                self._save_checkpoint(epoch)
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
        attn_dropout=0.1,           # transformer encoder dropout
        mlp_dropout=0.4,            # MLP head starting dropout (decreases: 0.4 → 0.3 → 0.2 → 0.1)
        lr_scheduler="cosine_warmup",    # "constant" | "cosine_warmup" | "step"
        warmup_epochs=5,            # used by cosine_warmup
        # step_size=10,             # used by step
        # step_gamma=0.5,           # used by step
        weighted_loss=False,        # weight CE loss by inverse class frequency (to counter class imbalance)
    )
    trainer.train()
