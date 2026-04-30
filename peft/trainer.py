"""
PEFT Trainer. AMP + grad accumulation + best-AUC checkpointing.

See peft/IMPLEMENTATION_PLAN.md §5 Step 4.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.amp import GradScaler, autocast
from tqdm import tqdm


class PEFTTrainer:
    def __init__(
        self,
        config: dict,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object],
        logger,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        amp_dtype = config.get("clip", {}).get("amp_dtype", "fp16")
        self.amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                          "fp32": torch.float32}[amp_dtype]
        self.amp_enabled = self.amp_dtype != torch.float32

        # GradScaler is a no-op for bf16/fp32; harmless to keep around.
        self.scaler = GradScaler(self.device.type, enabled=(self.amp_dtype == torch.float16))

        self.grad_accum = int(config.get("grad_accum_steps", 1))
        self.best_auc = 0.0

    def _autocast(self):
        return autocast(self.device.type,
                        dtype=self.amp_dtype,
                        enabled=self.amp_enabled)

    def _log_cuda_memory(self, prefix: str):
        if self.device.type != "cuda":
            return
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        self.logger.info(
            f"{prefix} cuda_mem allocated={allocated:.2f}GB "
            f"reserved={reserved:.2f}GB peak={peak:.2f}GB"
        )

    def train_epoch(self, epoch: int, train_loader, val_loader) -> float:
        self.model.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.optimizer.zero_grad(set_to_none=True)
        running = 0.0
        n_batches = len(train_loader)
        n_epochs = int(self.config.get("num_epochs", "?"))
        pbar = tqdm(train_loader, desc=f"ep{epoch}", unit="batch", leave=False)

        for i, (x, y) in enumerate(pbar):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            with self._autocast():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y) / self.grad_accum

            self.scaler.scale(loss).backward()

            step_now = ((i + 1) % self.grad_accum == 0) or (i + 1 == n_batches)
            if step_now:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            running += float(loss.item()) * self.grad_accum
            pbar.set_postfix(loss=f"{loss.item() * self.grad_accum:.4f}")
            if i == 0 or (i + 1) % 10 == 0:
                self._log_cuda_memory(f"epoch {epoch} batch {i + 1}/{n_batches}")

        avg_loss = running / max(n_batches, 1)
        auc, acc, acc_best, best_thresh = self.eval_epoch(val_loader)
        self._log_cuda_memory(f"epoch {epoch} after val")

        lr = self.optimizer.param_groups[0]["lr"]
        is_best = auc > self.best_auc
        flag = "  ★ new best" if is_best else ""
        self.logger.info(
            f"epoch {epoch}/{n_epochs}  "
            f"loss={avg_loss:.4f}  "
            f"val_auc={auc:.4f}  val_acc={acc:.4f}  val_acc@best={acc_best:.4f}  "
            f"thr={best_thresh:.3f}  lr={lr:.2e}"
            f"{flag}"
        )
        return auc

    @torch.no_grad()
    def eval_epoch(self, val_loader) -> tuple[float, float, float, float]:
        self.model.eval()
        all_probs, all_labels = [], []

        for x, y in val_loader:
            x = x.to(self.device, non_blocking=True)
            with self._autocast():
                logits = self.model(x)
            probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu()
            all_probs.append(probs)
            all_labels.append(y)

        probs = torch.cat(all_probs).numpy()
        labels = torch.cat(all_labels).numpy()

        try:
            auc = float(roc_auc_score(labels, probs))
        except ValueError:
            auc = float("nan")

        acc = float(((probs >= 0.5) == labels).mean())
        try:
            fpr, tpr, thresholds = roc_curve(labels, probs)
            best_thresh = float(thresholds[(tpr - fpr).argmax()])
            acc_best = float(((probs >= best_thresh) == labels).mean())
        except ValueError:
            best_thresh = float("nan")
            acc_best = float("nan")

        return auc, acc, acc_best, best_thresh

    def save_best(self, auc: float, out_path: str) -> bool:
        if auc > self.best_auc:
            self.best_auc = auc
            torch.save(self.model.trainable_state_dict(), out_path)
            return True
        return False

    def save_checkpoint(self, epoch: int, out_path: str,
                        per_epoch_val_auc: list) -> None:
        """Full training state for resumption. Overwritten every epoch."""
        payload = {
            "epoch":               epoch,
            "model_state_dict":    self.model.trainable_state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
                                    if self.scheduler is not None else None,
            "scaler_state_dict":   self.scaler.state_dict(),
            "best_auc":            self.best_auc,
            "per_epoch_val_auc":   per_epoch_val_auc,
        }
        torch.save(payload, out_path)

    def load_checkpoint(self, path: str) -> tuple[int, list]:
        """Load resume checkpoint. Returns (start_epoch, per_epoch_val_auc)."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_trainable_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler is not None and ckpt.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.best_auc = ckpt["best_auc"]
        start_epoch = ckpt["epoch"] + 1
        self.logger.info(
            f"resumed from epoch {ckpt['epoch']} (best_auc={self.best_auc:.4f})"
        )
        return start_epoch, ckpt.get("per_epoch_val_auc", [])

    def save_run_config(self, path: str, metrics: Optional[dict] = None) -> None:
        payload = {
            "saved_utc":      datetime.now(timezone.utc).isoformat(),
            "config":         self.config,
            "best_val_auc":   self.best_auc,
            "metrics":        metrics or {},
            "device":         str(self.device),
            "amp_dtype":      str(self.amp_dtype).replace("torch.", ""),
            "grad_accum":     self.grad_accum,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.logger.info(f"wrote run_config to {path}")
