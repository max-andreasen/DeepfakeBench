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
        self.last_epoch_metrics = {}
        loss_cfg = config.get("loss", {})
        self.ua_enabled = bool(loss_cfg.get("ua_enabled", False))
        self.alignment_weight = float(loss_cfg.get("alignment_weight", 0.1))
        self.uniformity_weight = float(loss_cfg.get("uniformity_weight", 0.5))

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

    def _ua_losses(self, features: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniformity/alignment losses on normalized per-frame features.

        `features` is [B, T, D] and is already L2-normalized by CompositePEFT.
        Labels are video-level, so each frame inherits its video's binary label.
        """
        z = features.reshape(-1, features.shape[-1])
        y = labels.repeat_interleave(features.shape[1])

        zero = z.new_zeros(())
        if z.shape[0] < 2:
            return zero, zero

        pairwise_sq = torch.cdist(z, z, p=2).pow(2)
        same_class = y[:, None].eq(y[None, :])
        not_self = ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
        positive_mask = same_class & not_self
        alignment = pairwise_sq[positive_mask].mean() if positive_mask.any() else zero

        upper = torch.triu(torch.ones_like(pairwise_sq, dtype=torch.bool), diagonal=1)
        uniformity = torch.log(torch.exp(-2.0 * pairwise_sq[upper]).mean().clamp_min(1e-12))
        return alignment, uniformity

    def train_epoch(self, epoch: int, train_loader, val_loader) -> float:
        self.model.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.optimizer.zero_grad(set_to_none=True)
        running = 0.0
        running_ce = 0.0
        running_align = 0.0
        running_uniform = 0.0
        n_batches = len(train_loader)
        n_epochs = int(self.config.get("num_epochs", "?"))
        pbar = tqdm(train_loader, desc=f"ep{epoch}", unit="batch", leave=False)

        for i, (x, y) in enumerate(pbar):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            if i == 0 or (i + 1) % 10 == 0:
                self._log_cuda_memory(f"epoch {epoch} batch {i + 1}/{n_batches} before forward")

            try:
                with self._autocast():
                    if self.ua_enabled:
                        logits, features = self.model(x, return_features=True)
                    else:
                        logits = self.model(x)
                        features = None

                ce_loss = F.cross_entropy(logits.float(), y)
                align_loss = logits.new_zeros(())
                uniform_loss = logits.new_zeros(())
                if self.ua_enabled and (self.alignment_weight or self.uniformity_weight):
                    align_loss, uniform_loss = self._ua_losses(features.float(), y)
                total_loss = (
                    ce_loss
                    + self.alignment_weight * align_loss
                    + self.uniformity_weight * uniform_loss
                )
                loss = total_loss / self.grad_accum

                self.scaler.scale(loss).backward()
            except RuntimeError:
                self._log_cuda_memory(f"epoch {epoch} batch {i + 1}/{n_batches} failed")
                raise

            step_now = ((i + 1) % self.grad_accum == 0) or (i + 1 == n_batches)
            if step_now:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            running += float(total_loss.detach().item())
            running_ce += float(ce_loss.detach().item())
            running_align += float(align_loss.detach().item())
            running_uniform += float(uniform_loss.detach().item())
            pbar.set_postfix(loss=f"{total_loss.detach().item():.4f}")
            if i == 0 or (i + 1) % 10 == 0:
                self._log_cuda_memory(f"epoch {epoch} batch {i + 1}/{n_batches} after backward")

        avg_loss = running / max(n_batches, 1)
        avg_ce = running_ce / max(n_batches, 1)
        avg_align = running_align / max(n_batches, 1)
        avg_uniform = running_uniform / max(n_batches, 1)
        auc, acc, acc_best, best_thresh, val_loss = self.eval_epoch(val_loader)
        self._log_cuda_memory(f"epoch {epoch} after val")

        lr = self.optimizer.param_groups[0]["lr"]
        is_best = auc > self.best_auc
        flag = "  ★ new best" if is_best else ""
        ua_log = (
            f"ce={avg_ce:.4f}  align={avg_align:.4f}  uniform={avg_uniform:.4f}  "
            if self.ua_enabled else ""
        )
        self.logger.info(
            f"epoch {epoch}/{n_epochs}  "
            f"loss={avg_loss:.4f}  "
            f"{ua_log}"
            f"val_loss={val_loss:.4f}  "
            f"val_auc={auc:.4f}  val_acc={acc:.4f}  val_acc@best={acc_best:.4f}  "
            f"thr={best_thresh:.3f}  lr={lr:.2e}"
            f"{flag}"
        )
        self.last_epoch_metrics = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_ce": avg_ce,
            "train_align": avg_align,
            "train_uniform": avg_uniform,
            "val_loss": val_loss,
            "val_auc": auc,
            "val_acc": acc,
            "val_acc_at_best": acc_best,
            "best_threshold": best_thresh,
            "lr": lr,
        }
        return auc

    @torch.no_grad()
    def eval_epoch(self, val_loader) -> tuple[float, float, float, float, float]:
        self.model.eval()
        all_probs, all_labels = [], []
        loss_sum = 0.0
        n_examples = 0

        for x, y in val_loader:
            x = x.to(self.device, non_blocking=True)
            y_device = y.to(self.device, non_blocking=True)
            with self._autocast():
                logits = self.model(x)
            loss_sum += float(F.cross_entropy(logits.float(), y_device, reduction="sum").item())
            n_examples += int(y_device.numel())
            probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu()
            all_probs.append(probs)
            all_labels.append(y)

        probs = torch.cat(all_probs).numpy()
        labels = torch.cat(all_labels).numpy()
        val_loss = loss_sum / max(n_examples, 1)

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

        return auc, acc, acc_best, best_thresh, val_loss

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
