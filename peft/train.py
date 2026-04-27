"""
PEFT training entry point.

Usage:
    python peft/train.py --config peft/configs/peft_ff_mtcnn.yaml
    python peft/train.py --config peft/configs/peft_smoke.yaml \
                        --max_videos 20 --num_epochs 1

See peft/IMPLEMENTATION_PLAN.md §5 Step 5.
"""

import argparse
import datetime as dt
import os
import random
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from logger import create_logger                          # noqa: E402
from peft.data_loader import FramePEFTDataset             # noqa: E402
from peft.models.clip_peft import CompositePEFT           # noqa: E402
from peft.trainer import PEFTTrainer                      # noqa: E402


def init_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_log_path(cfg: dict) -> Path:
    timenow = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = Path(cfg["log_dir"])
    if not log_dir.is_absolute():
        log_dir = REPO_ROOT / log_dir
    return log_dir / f"{cfg['tag']}_{timenow}"


def build_optimizer(cfg: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Important: filter to trainable params only — full param list would
    create AdamW state for 300 M frozen weights and waste ~2.4 GB VRAM."""
    opt_cfg = cfg["optimizer"]
    name = opt_cfg["type"]
    params = model.trainable_parameters()
    if name == "adamw":
        return AdamW(
            params,
            lr=float(opt_cfg["lr"]),
            betas=(float(opt_cfg["beta1"]), float(opt_cfg["beta2"])),
            eps=float(opt_cfg["eps"]),
            weight_decay=float(opt_cfg["weight_decay"]),
        )
    raise NotImplementedError(f"optimizer {name} not implemented")


def build_scheduler(cfg: dict, optimizer: torch.optim.Optimizer):
    sched = cfg.get("lr_scheduler")
    n_epochs = int(cfg["num_epochs"])
    if sched in (None, "constant"):
        return None
    if sched == "cosine":
        return CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0)
    if sched == "cosine_warmup":
        warm = int(cfg.get("warmup_epochs", 0))
        warmup = LinearLR(optimizer, start_factor=1e-3, total_iters=warm)
        cosine = CosineAnnealingLR(optimizer, T_max=n_epochs - warm)
        return SequentialLR(optimizer, [warmup, cosine], milestones=[warm])
    raise NotImplementedError(f"scheduler {sched} not implemented")


def build_loaders(cfg: dict, preprocess):
    root = Path(cfg.get("root_dir", "")) if cfg.get("root_dir") else REPO_ROOT
    split_file     = str(root / cfg["split_file"])
    rearrange_json = str(root / cfg["rearrange_json"])
    max_videos     = cfg.get("max_videos")

    train_ds = FramePEFTDataset(
        split_file=split_file,
        rearrange_json=rearrange_json,
        dataset_name=cfg["dataset_name"],
        split="train",
        num_frames=cfg["num_frames"]["train"],
        preprocess=preprocess,
        max_videos=max_videos,
    )
    val_ds = FramePEFTDataset(
        split_file=split_file,
        rearrange_json=rearrange_json,
        dataset_name=cfg["dataset_name"],
        split="val",
        num_frames=cfg["num_frames"]["val"],
        preprocess=preprocess,
        max_videos=max_videos,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batchSize"]["train"],
        shuffle=True,
        num_workers=int(cfg["workers"]),
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batchSize"]["val"],
        shuffle=False,
        num_workers=int(cfg["workers"]),
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def main():
    ap = argparse.ArgumentParser(description="PEFT (LN-tuned CLIP + temporal head) training.")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to a previous run dir to resume from (loads checkpoint.pth).")
    ap.add_argument("--max_videos", type=int, default=None,
                    help="Override config max_videos (smoke runs).")
    ap.add_argument("--num_epochs", type=int, default=None,
                    help="Override config num_epochs (smoke runs).")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.max_videos is not None:
        cfg["max_videos"] = args.max_videos
    if args.num_epochs is not None:
        cfg["num_epochs"] = args.num_epochs

    init_seed(int(cfg["seed"]))

    resume_dir = Path(args.resume) if args.resume else None
    if resume_dir and not resume_dir.is_absolute():
        resume_dir = REPO_ROOT / resume_dir

    # Reuse the existing run dir when resuming; create a new one otherwise.
    if resume_dir:
        log_path = resume_dir
    else:
        log_path = build_log_path(cfg)
    log_path.mkdir(parents=True, exist_ok=True)
    logger = create_logger(str(log_path / "training.log"))
    logger.info("=" * 60)
    logger.info(f"Config: {cfg}")
    logger.info(f"Log dir: {log_path}")

    # CLIP model + preprocess transform from a single open_clip call. The
    # preprocess transform is what the dataset uses; the model is moved into
    # CompositePEFT.
    logger.info("Loading CLIP backbone...")
    _, _, preprocess = open_clip.create_model_and_transforms(
        cfg["clip"]["name"], pretrained=cfg["clip"]["pretrained"]
    )

    train_loader, val_loader = build_loaders(cfg, preprocess)
    logger.info(f"train batches: {len(train_loader)}  val batches: {len(val_loader)}")

    model = CompositePEFT(
        clip_name=cfg["clip"]["name"],
        clip_pretrained=cfg["clip"]["pretrained"],
        ln_scope=cfg["clip"].get("ln_scope", "all"),
        grad_checkpointing=bool(cfg["clip"].get("grad_checkpointing", True)),
        temporal_kwargs=cfg.get("temporal", {}),
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"trainable={n_train/1e6:.2f}M total={n_total/1e6:.1f}M")

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    trainer = PEFTTrainer(cfg, model, optimizer, scheduler, logger)

    ckpt_path = log_path / "model.pth"
    run_cfg_path = log_path / "run_config.json"

    save_ckpt = bool(cfg.get("save_ckpt", True))
    resume_ckpt_path = log_path / "checkpoint.pth"
    completed = False

    # Load resume state (metrics history + model/optimizer/scheduler).
    start_epoch = 0
    metrics = {"per_epoch_val_auc": []}
    if resume_dir and resume_ckpt_path.exists():
        start_epoch, prior_aucs = trainer.load_checkpoint(str(resume_ckpt_path))
        metrics["per_epoch_val_auc"] = prior_aucs
    elif resume_dir:
        logger.warning(f"--resume set but no checkpoint.pth found in {resume_dir}; starting fresh")

    try:
        for epoch in range(start_epoch, int(cfg["num_epochs"])):
            auc = trainer.train_epoch(epoch, train_loader, val_loader)
            metrics["per_epoch_val_auc"].append(auc)
            if save_ckpt:
                trainer.save_best(auc, str(ckpt_path))
                trainer.save_checkpoint(epoch, str(resume_ckpt_path),
                                        metrics["per_epoch_val_auc"])
            if scheduler is not None:
                scheduler.step()
        completed = True
    finally:
        # Fallback: if save_best never triggered (e.g. smoke run with single-class
        # val producing nan AUC), save the final weights so there's always a checkpoint.
        if save_ckpt and not ckpt_path.exists():
            torch.save(model.trainable_state_dict(), str(ckpt_path))
            logger.info(f"saved final ckpt (no best found) to {ckpt_path}")
        metrics["completed"] = completed
        trainer.save_run_config(str(run_cfg_path), metrics=metrics)
        logger.info(f"best_val_auc={trainer.best_auc:.4f}")
        logger.info("training complete." if completed
                    else "training interrupted before completion.")


if __name__ == "__main__":
    main()
