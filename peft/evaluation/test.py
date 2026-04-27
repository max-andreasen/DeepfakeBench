"""
PEFT evaluation entry point.

Usage:
    python peft/evaluation/test.py --config peft/configs/peft_eval_ff_test.yaml
    python peft/evaluation/test.py --config peft/configs/peft_eval_cdfv2.yaml \\
        --run_tag custom_tag

See peft/IMPLEMENTATION_PLAN.md §5 Step 8 / Step 9.
"""

import argparse
import json
import sys
from pathlib import Path

import open_clip
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from logger import create_logger                          # noqa: E402
from peft.data_loader import FramePEFTTestDataset         # noqa: E402
from peft.evaluation.tester import PEFTTester             # noqa: E402
from peft.models.clip_peft import CompositePEFT           # noqa: E402


REQUIRED_FIELDS = [
    "trained_model_dir",
    "output_dir",
    "split_file",
    "rearrange_json",
    "dataset_name",
    "split",
    "num_frames",
    "batch_size",
    "device",
    "seed",
]


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate a PEFT checkpoint.")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--run_tag", type=str, default=None,
                    help="output subfolder under <output_dir>/<trained_dir>/. "
                         "defaults to the config filename stem.")
    args = ap.parse_args()
    if args.run_tag is None:
        args.run_tag = Path(args.config).stem
    return args


def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in REQUIRED_FIELDS if k not in cfg]
    if missing:
        raise ValueError(f"Eval config missing required keys: {missing}")
    cfg.setdefault("workers", 4)
    cfg.setdefault("window_aggregation", "mean")
    cfg.setdefault("softmax_temp", 1.0)
    cfg.setdefault("root_dir", str(REPO_ROOT))
    return cfg


def load_trained_peft_model(trained_model_dir: str, root_dir: str):
    """Rebuild CompositePEFT from the training run's run_config.json and load
    the trainable state dict from model.pth."""
    base = Path(trained_model_dir)
    if not base.is_absolute():
        base = Path(root_dir) / base

    run_cfg_path = base / "run_config.json"
    model_path = base / "model.pth"
    if not run_cfg_path.exists():
        raise FileNotFoundError(f"missing {run_cfg_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"missing {model_path}")

    with open(run_cfg_path) as f:
        run_cfg = json.load(f)
    train_cfg = run_cfg["config"]

    model = CompositePEFT(
        clip_name=train_cfg["clip"]["name"],
        clip_pretrained=train_cfg["clip"]["pretrained"],
        ln_scope=train_cfg["clip"].get("ln_scope", "all"),
        grad_checkpointing=False,                  # eval is no_grad — checkpointing wastes time
        temporal_kwargs=train_cfg.get("temporal", {}),
    )
    state = torch.load(str(model_path), map_location="cpu")
    model.load_trainable_state_dict(state)
    return model, run_cfg


def setup_output(cfg, run_tag):
    trained_dir_basename = Path(cfg["trained_model_dir"]).name
    out_dir = Path(cfg["output_dir"]) / trained_dir_basename / run_tag
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(str(out_dir / "test.log"))
    return out_dir, logger


def main():
    args = parse_args()
    cfg = load_config(args.config)

    out_dir, logger = setup_output(cfg, args.run_tag)
    logger.info(f"Eval config: {cfg}")
    logger.info(f"Output dir: {out_dir}")

    model, run_cfg = load_trained_peft_model(cfg["trained_model_dir"], cfg["root_dir"])
    logger.info(f"Loaded checkpoint from {cfg['trained_model_dir']}")
    logger.info(f"Trained on dataset: {run_cfg['config']['dataset_name']}")
    logger.info(f"Best val AUC during training: {run_cfg.get('best_val_auc')}")

    # Get the same preprocess transform CLIP was loaded with at train time.
    _, _, preprocess = open_clip.create_model_and_transforms(
        run_cfg["config"]["clip"]["name"],
        pretrained=run_cfg["config"]["clip"]["pretrained"],
    )

    root = Path(cfg["root_dir"])
    test_ds = FramePEFTTestDataset(
        split_file=str(root / cfg["split_file"]),
        rearrange_json=str(root / cfg["rearrange_json"]),
        dataset_name=cfg["dataset_name"],
        split=cfg["split"],
        num_frames=int(cfg["num_frames"]),
        preprocess=preprocess,
    )
    loader = DataLoader(
        test_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["workers"]),
        pin_memory=True,
    )
    logger.info(
        f"test videos: {len(test_ds)}  windows/video: {test_ds.n_windows}  "
        f"total batches: {len(loader)}"
    )

    # Pass the trained-side `clip` block through so PEFTTester can read amp_dtype.
    tester_cfg = dict(cfg)
    tester_cfg["clip"] = run_cfg["config"]["clip"]
    tester = PEFTTester(tester_cfg, model, logger)

    torch.manual_seed(int(cfg["seed"]))
    standard = tester.evaluate(loader, shuffle_frames=False)

    shuffled = None
    if bool(cfg.get("eval_shuffled", False)):
        torch.manual_seed(int(cfg["seed"]))   # reproducible shuffle pass
        shuffled = tester.evaluate(loader, shuffle_frames=True)

    tester.save_results(out_dir, standard, eval_config_dict=cfg, shuffled=shuffled)


if __name__ == "__main__":
    main()
