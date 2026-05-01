"""
Optuna launcher for PEFT pilot searches.

This runs the normal PEFT training stack directly so Optuna can observe
per-epoch validation AUC and prune underperforming trials.

Example:
    python peft/optuna_search.py \
        --search-config peft/search_configs/peft_gend_pilot12.yaml
"""

import argparse
import copy
import csv
import gc
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import open_clip
import optuna
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from logger import create_logger  # noqa: E402
from peft.models.clip_peft import CompositePEFT  # noqa: E402
from peft.train import (  # noqa: E402
    build_loaders,
    build_optimizer,
    build_scheduler,
    init_seed,
)
from peft.trainer import PEFTTrainer  # noqa: E402


ANCHOR_COMMON = {
    "clip.feature_layer": "pre_proj",
    "optimizer.lr": 2.0e-5,
    "optimizer.weight_decay": 1.0e-4,
    "temporal.num_layers": 1,
    "temporal.dim_feedforward": 1024,
    "temporal.attn_dropout": 0.0,
    "temporal.mlp_dropout": 0.25,
    "temporal.mlp_hidden_dim": 256,
}


DEFAULT_SEARCH_CONFIG = {
    "base_config": "peft/configs/peft_ff_cdfv2val.yaml",
    "study_name": "peft_gend_pilot12",
    "n_trials": 12,
    "epochs": 5,
    "storage": "peft/searches/peft_optuna.db",
    "output_dir": "peft/searches/runs",
    "sampler": {
        "type": "tpe",
        "seed": 1024,
        "startup_trials": 4,
    },
    "pruner": {
        "type": "median",
        "warmup_epochs": 2,
    },
    "anchors": {
        "enabled": True,
    },
}


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_search_config(path: Path) -> dict:
    cfg = deep_merge(DEFAULT_SEARCH_CONFIG, load_yaml(path))
    required = ["base_config", "study_name", "n_trials", "epochs", "storage", "output_dir"]
    missing = [key for key in required if cfg.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Search config {path} is missing required keys: {missing}")
    return cfg


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    cursor = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def storage_url(storage: str) -> str:
    if "://" in storage:
        return storage
    path = Path(storage)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def setup_study_logger(study_dir: Path) -> logging.Logger:
    logger = logging.getLogger("peft_optuna_search")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(str(study_dir / "study.log"))
    handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    optuna_logger = optuna.logging.get_logger("optuna")
    for handler in list(optuna_logger.handlers):
        if getattr(handler, "_peft_optuna_file", False):
            optuna_logger.removeHandler(handler)
            handler.close()
    optuna_handler = logging.FileHandler(str(study_dir / "optuna_internal.log"))
    optuna_handler._peft_optuna_file = True
    optuna_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    optuna_logger.addHandler(optuna_handler)
    return logger


def sample_param(trial: optuna.Trial, name: str, spec: dict) -> Any:
    kind = spec["type"]
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if kind == "float":
        low, high = spec["range"]
        return trial.suggest_float(name, float(low), float(high), log=bool(spec.get("log", False)))
    if kind == "int":
        low, high = spec["range"]
        step = int(spec.get("step", 1))
        return trial.suggest_int(name, int(low), int(high), step=step, log=bool(spec.get("log", False)))
    raise ValueError(f"Unsupported search-space type for {name}: {kind}")


def sample_overrides(trial: optuna.Trial, search_space: dict) -> dict:
    return {name: sample_param(trial, name, spec) for name, spec in search_space.items()}


def anchor_trials(search_space: dict) -> list[dict]:
    anchors = []
    for l2_norm in (True, False):
        for ua_enabled in (True, False):
            trial = dict(ANCHOR_COMMON)
            trial["clip.l2_normalize_features"] = l2_norm
            trial["loss.ua_enabled"] = ua_enabled
            anchors.append({k: v for k, v in trial.items() if k in search_space})
    return anchors


def apply_trial_config(base_cfg: dict, overrides: dict, trial_number: int, epochs: int) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg.pop("search_space", None)
    for key, value in overrides.items():
        set_dotted(cfg, key, value)

    cfg["num_epochs"] = int(epochs)
    cfg["seed"] = int(base_cfg.get("seed", 1024)) + int(trial_number)
    cfg["tag"] = f"{base_cfg.get('tag', 'peft')}_trial_{trial_number:04d}"
    return cfg


def build_model(cfg: dict) -> CompositePEFT:
    return CompositePEFT(
        clip_name=cfg["clip"]["name"],
        clip_pretrained=cfg["clip"]["pretrained"],
        ln_scope=cfg["clip"].get("ln_scope", "all"),
        feature_layer=cfg["clip"].get("feature_layer", "pre_proj"),
        l2_normalize_features=bool(cfg["clip"].get("l2_normalize_features", False)),
        grad_checkpointing=bool(cfg["clip"].get("grad_checkpointing", True)),
        temporal_kwargs=cfg.get("temporal", {}),
    )


def write_trial_params(trial_dir: Path, trial: optuna.Trial, overrides: dict, cfg: dict) -> None:
    payload = {
        "trial_number": trial.number,
        "overrides": overrides,
        "config": cfg,
    }
    with open(trial_dir / "trial_params.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_epoch_metrics(path: Path, row: dict) -> None:
    fields = [
        "epoch",
        "train_loss",
        "train_ce",
        "train_align",
        "train_uniform",
        "val_loss",
        "val_auc",
        "val_acc",
        "val_acc_at_best",
        "best_threshold",
        "lr",
    ]
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fields})


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_trial(trial: optuna.Trial, base_cfg: dict, study_dir: Path, epochs: int) -> float:
    search_space = base_cfg.get("search_space", {})
    overrides = sample_overrides(trial, search_space)
    cfg = apply_trial_config(base_cfg, overrides, trial.number, epochs)

    trial_dir = study_dir / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(trial_dir / "trial_config.yaml", cfg)
    write_trial_params(trial_dir, trial, overrides, cfg)
    trial.set_user_attr("trial_dir", str(trial_dir))
    trial.set_user_attr("overrides", overrides)

    logger = create_logger(str(trial_dir / "training.log"))
    logger.info("=" * 60)
    logger.info(f"Optuna trial {trial.number}")
    logger.info(f"Overrides: {overrides}")
    logger.info(f"Config: {cfg}")
    logger.info(f"Log dir: {trial_dir}")

    init_seed(int(cfg["seed"]))

    logger.info("Loading CLIP preprocess...")
    _, _, preprocess = open_clip.create_model_and_transforms(
        cfg["clip"]["name"], pretrained=cfg["clip"]["pretrained"]
    )
    train_loader, val_loader = build_loaders(cfg, preprocess)
    logger.info(f"train batches: {len(train_loader)}  val batches: {len(val_loader)}")

    logger.info("Loading PEFT model...")
    model = build_model(cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"trainable={n_train/1e6:.2f}M total={n_total/1e6:.1f}M")

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    trainer = PEFTTrainer(cfg, model, optimizer, scheduler, logger)

    ckpt_path = trial_dir / "model.pth"
    checkpoint_path = trial_dir / "checkpoint.pth"
    run_cfg_path = trial_dir / "run_config.json"
    metrics = {
        "per_epoch": [],
        "per_epoch_val_auc": [],
        "per_epoch_train_loss": [],
        "per_epoch_val_loss": [],
        "optuna_trial_number": trial.number,
        "optuna_overrides": overrides,
        "completed": False,
        "pruned": False,
    }
    epoch_metrics_path = trial_dir / "epoch_metrics.csv"

    try:
        for epoch in range(int(cfg["num_epochs"])):
            auc = trainer.train_epoch(epoch, train_loader, val_loader)
            epoch_metrics = dict(trainer.last_epoch_metrics)
            metrics["per_epoch_val_auc"].append(auc)
            metrics["per_epoch_train_loss"].append(epoch_metrics.get("train_loss"))
            metrics["per_epoch_val_loss"].append(epoch_metrics.get("val_loss"))
            metrics["per_epoch"].append(epoch_metrics)
            append_epoch_metrics(epoch_metrics_path, epoch_metrics)

            value = auc if math.isfinite(auc) else 0.0
            trial.report(value, step=epoch)
            trial.set_user_attr("epoch_metrics", metrics["per_epoch"])
            logger.info(
                f"optuna_report trial={trial.number} epoch={epoch} "
                f"value={value:.4f} params={overrides}"
            )

            if bool(cfg.get("save_ckpt", True)):
                trainer.save_best(auc, str(ckpt_path))
                trainer.save_checkpoint(epoch, str(checkpoint_path), metrics["per_epoch_val_auc"])

            if scheduler is not None:
                scheduler.step()

            if trial.should_prune():
                metrics["pruned"] = True
                raise optuna.TrialPruned(f"pruned after epoch {epoch} with val_auc={auc:.4f}")

        metrics["completed"] = True
        trial.set_user_attr("best_val_auc", trainer.best_auc)
        return trainer.best_auc
    finally:
        if bool(cfg.get("save_ckpt", True)) and not ckpt_path.exists():
            torch.save(model.trainable_state_dict(), str(ckpt_path))
            logger.info(f"saved final ckpt (no best found) to {ckpt_path}")
        trainer.save_run_config(str(run_cfg_path), metrics=metrics)
        logger.info(f"trial_best_val_auc={trainer.best_auc:.4f}")
        free_gpu()


def write_study_manifest(
    study_dir: Path,
    search_config_path: Path,
    search_cfg: dict,
    base_cfg: dict,
    anchors: list[dict],
) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "study_name": search_cfg["study_name"],
        "search_config": str(search_config_path.resolve()),
        "base_config": str((REPO_ROOT / search_cfg["base_config"]).resolve()),
        "search_config_values": search_cfg,
        "sampler": search_cfg["sampler"]["type"],
        "pruner": search_cfg["pruner"]["type"],
        "search_space": base_cfg.get("search_space", {}),
        "anchors": anchors,
    }
    with open(study_dir / "study_manifest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_summary(study: optuna.Study, study_dir: Path) -> None:
    study.trials_dataframe().to_csv(study_dir / "all_trials.csv", index=False)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    completed.sort(key=lambda t: (t.value if t.value is not None else -float("inf")), reverse=True)
    wall_s = sum((t.duration.total_seconds() for t in study.trials if t.duration is not None), 0.0)

    try:
        best = study.best_trial
    except ValueError:
        best = None

    lines = [f"# Study: {study.study_name}", ""]
    lines.append(f"- Total trials: {len(study.trials)}")
    lines.append(f"- Completed: {len(completed)}")
    lines.append(f"- Pruned: {len(pruned)}")
    lines.append(f"- Failed: {len(failed)}")
    lines.append(f"- Wall-clock: {wall_s/3600:.2f} h")
    if best is not None:
        lines.append(f"- Best trial: {best.number}")
        lines.append(f"- Best val_auc: {best.value:.4f}")
    lines.extend(["", "## Top 10", "", "| Rank | Trial | Value | Params |", "| --- | --- | --- | --- |"])
    for rank, trial in enumerate(completed[:10], 1):
        params_str = ", ".join(f"{k}={v}" for k, v in trial.params.items())
        lines.append(f"| {rank} | {trial.number} | {trial.value:.4f} | {params_str} |")
    with open(study_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    if best is not None:
        best_run = study_dir / f"trial_{best.number:04d}" / "run_config.json"
        if best_run.exists():
            with open(best_run, "r", encoding="utf-8") as f:
                payload = json.load(f)
            with open(study_dir / "best_config.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)


def write_optuna_plots(study: optuna.Study, study_dir: Path) -> None:
    try:
        import optuna.visualization as vis
    except ImportError:
        return

    plots_dir = study_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_specs = [
        ("optimization_history", vis.plot_optimization_history),
        ("param_importances", vis.plot_param_importances),
        ("parallel_coordinate", vis.plot_parallel_coordinate),
        ("slice_plot", vis.plot_slice),
    ]
    for name, fn in plot_specs:
        try:
            fn(study).write_html(str(plots_dir / f"{name}.html"))
        except Exception as exc:
            with open(plots_dir / f"{name}.error.txt", "w", encoding="utf-8") as f:
                f.write(str(exc) + "\n")


def make_sampler(search_cfg: dict):
    sampler_cfg = search_cfg.get("sampler", {})
    sampler_type = sampler_cfg.get("type", "tpe")
    seed = int(sampler_cfg.get("seed", 1024))
    if sampler_type == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if sampler_type == "tpe":
        return optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=int(sampler_cfg.get("startup_trials", 4)),
            multivariate=True,
        )
    raise ValueError(f"Unknown sampler: {sampler_type}")


def make_pruner(search_cfg: dict):
    pruner_cfg = search_cfg.get("pruner", {})
    pruner_type = pruner_cfg.get("type", "median")
    if pruner_type == "none":
        return optuna.pruners.NopPruner()
    if pruner_type == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=int(search_cfg.get("sampler", {}).get("startup_trials", 4)),
            n_warmup_steps=int(pruner_cfg.get("warmup_epochs", 2)),
            interval_steps=1,
        )
    raise ValueError(f"Unknown pruner: {pruner_type}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a PEFT Optuna pilot search.")
    ap.add_argument(
        "--search-config",
        default="peft/search_configs/peft_gend_pilot12.yaml",
        help="YAML file containing Optuna study settings.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    search_config_path = Path(args.search_config)
    if not search_config_path.is_absolute():
        search_config_path = REPO_ROOT / search_config_path
    search_cfg = load_search_config(search_config_path)

    base_config_path = Path(search_cfg["base_config"])
    if not base_config_path.is_absolute():
        base_config_path = REPO_ROOT / base_config_path
    base_cfg = load_yaml(base_config_path)

    study_dir = Path(search_cfg["output_dir"])
    if not study_dir.is_absolute():
        study_dir = REPO_ROOT / study_dir
    study_dir = study_dir / search_cfg["study_name"]
    study_dir.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_logger = setup_study_logger(study_dir)

    sampler = make_sampler(search_cfg)
    pruner = make_pruner(search_cfg)
    study = optuna.create_study(
        study_name=search_cfg["study_name"],
        direction="maximize",
        storage=storage_url(search_cfg["storage"]),
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    anchors = (
        []
        if not search_cfg.get("anchors", {}).get("enabled", True)
        else anchor_trials(base_cfg.get("search_space", {}))
    )
    if anchors and len(study.trials) == 0:
        for params in anchors:
            study.enqueue_trial(params)

    write_study_manifest(study_dir, search_config_path, search_cfg, base_cfg, anchors)
    study_logger.info(f"=== Starting study '{search_cfg['study_name']}' ===")
    study_logger.info(f"Search config: {search_config_path}")
    study_logger.info(f"Search config values: {search_cfg}")
    study_logger.info(f"Base config: {base_config_path}")
    study_logger.info(f"Search space: {base_cfg.get('search_space', {})}")
    study_logger.info(f"Anchors: {anchors}")

    def trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        dur = trial.duration.total_seconds() if trial.duration is not None else 0.0
        value = f"{trial.value:.4f}" if trial.value is not None else "n/a"
        try:
            best_num = study.best_trial.number
            best_val = f"{study.best_value:.4f}"
        except ValueError:
            best_num, best_val = -1, "n/a"
        line = (
            f"[trial {trial.number:04d}] state={trial.state.name:9s} "
            f"value={value} dur={dur:6.1f}s best={best_val} (#{best_num}) "
            f"params={trial.params}"
        )
        print(line, flush=True)
        study_logger.info(line)

    study.optimize(
        lambda trial: train_trial(trial, base_cfg, study_dir, int(search_cfg["epochs"])),
        n_trials=int(search_cfg["n_trials"]),
        gc_after_trial=True,
        show_progress_bar=True,
        callbacks=[trial_callback],
    )
    write_summary(study, study_dir)
    write_optuna_plots(study, study_dir)
    study_logger.info(f"=== Study '{search_cfg['study_name']}' complete ===")
    print(f"Study written to {study_dir}")


if __name__ == "__main__":
    main()
