"""
Config-driven CLIP embedding pipeline with probing support.

Run:
    python clip/embed.py --config clip/configs/config_name.yaml
    python clip/embed.py --config ... --max_videos 20 --device cuda:1
    python clip/embed.py --config ... -r                # resume a crashed run

Assumes input is pre-extracted, face-aligned frames (PNGs from preprocess.py).

Layout (per run):
    <output_base_dir>/<safe_model_name>/
        run_config.json                   # resolved config + completion state
        pre_proj/
            catalogue.csv
            <dataset>/<label_cat>/<video>.npz   # [T, D] L2-normalised
        final/        (if extract.include_final)
            ...
        block_12/     (one dir per entry in extract.blocks)
            ...

Owns everything outside the CLIP forward pass:
    - config load + validation
    - df construction from rearrange JSON
    - frame loading from PNG paths
    - per-video loop, per-layer npz save, per-layer catalogue.csv, run_config.json
"""

import argparse
import importlib.util
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]


# Load CLIP_embedder.py by absolute path to sidestep the `clip/` folder vs
# the `clip` PyPI package name collision.
def _load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_embedder_mod = _load_module_from_path(
    "local_clip_embedder",
    Path(__file__).resolve().parent / "CLIP_embedder.py",
)
CLIP_EMBEDDER = _embedder_mod.CLIP_EMBEDDER

_logger_mod = _load_module_from_path("_repo_logger", REPO_ROOT / "logger.py")
create_logger = _logger_mod.create_logger

# Module-level logger. Silent until create_logger(log_file) is called in main().
# Shared name "deepfakebench" with logger.py.
log = logging.getLogger("deepfakebench")


# Maps label category strings (from rearrange.py JSON) to numeric labels.
# Convention: 1 = real, 0 = fake.
LABEL_MAP = {
    "CelebDFv2_real": 1, "CelebDFv2_fake": 0,
    "CelebDFv1_real": 1, "CelebDFv1_fake": 0,
    "FF-real":        1, "FF-DF": 0, "FF-F2F": 0, "FF-FS": 0, "FF-NT": 0,
    "DFDCP_Real":     1, "DFDCP_FakeA": 0, "DFDCP_FakeB": 0,
    "DFDC_Real":      1, "DFDC_Fake": 0,
    "UADFV_Real":     1, "UADFV_Fake": 0,
    "DF_real":        1, "DF_fake": 0,
}


# --------------------------------------------------------------------------- #
# Config & path helpers                                                       #
# --------------------------------------------------------------------------- #

def _resolve_path(p: Union[str, Path]) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    required = ["dataset_name", "input_json", "output_base_dir", "model", "T", "seed", "device"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Config {config_path} missing required keys: {missing}")
    for k in ("name", "pretrained"):
        if k not in cfg["model"]:
            raise ValueError(f"Config {config_path} missing model.{k}")
    return cfg


def _parse_extract(cfg: dict) -> Tuple[List[int], bool]:
    """Parse cfg['extract'] → (blocks, include_final). pre_proj is always on."""
    ex = cfg.get("extract") or {}
    blocks_raw = ex.get("blocks") or []
    if not isinstance(blocks_raw, list) or not all(isinstance(b, int) for b in blocks_raw):
        raise ValueError(f"extract.blocks must be a list of ints, got: {blocks_raw!r}")
    include_final = bool(ex.get("include_final", True))
    return sorted(set(blocks_raw)), include_final


def _safe_model_name(model_name: str) -> str:
    return str(model_name).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _get_model_dir(base_dir: Path, model_name: str) -> Path:
    """Parent dir for one model's outputs: <base_dir>/<safe_model_name>/"""
    d = base_dir / _safe_model_name(model_name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _layer_dir(model_dir: Path, layer_name: str) -> Path:
    d = model_dir / layer_name
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Input df                                                                    #
# --------------------------------------------------------------------------- #

def build_df_from_repo_json(json_path: Path, dataset_name: str) -> pd.DataFrame:
    """
    Reads a JSON file produced by rearrange.py and returns a DataFrame with:
        video_id, label, label_cat, frame_paths, dataset

    Splits are not tracked here; they live in datasets/splits/*.csv.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if dataset_name not in data:
        raise ValueError(f"'{dataset_name}' not found in {json_path}. Keys: {list(data.keys())}")

    rows = []
    for label_cat, splits in data[dataset_name].items():
        numeric_label = LABEL_MAP.get(label_cat)
        if numeric_label is None:
            raise ValueError(
                f"No label mapping for '{label_cat}'. Add it to LABEL_MAP in embed.py."
            )
        for videos_or_comp in splits.values():
            # FF++ JSONs have an extra compression level: split → c23 → videos
            first_val = next(iter(videos_or_comp.values()), None)
            if first_val is not None and isinstance(first_val, dict) and "frames" not in first_val:
                videos = {}
                for _comp, comp_videos in videos_or_comp.items():
                    videos.update(comp_videos)
            else:
                videos = videos_or_comp

            for video_id, video_info in videos.items():
                if "frames" not in video_info:
                    raise ValueError(
                        f"Video '{video_id}' in {json_path} has no 'frames' key. "
                        "This pipeline only supports pre-extracted frames."
                    )
                rows.append({
                    "dataset":     dataset_name,
                    "video_id":    video_id,
                    "label":       numeric_label,
                    "label_cat":   label_cat,
                    "frame_paths": sorted(video_info["frames"]),
                })

    df = pd.DataFrame(rows).drop_duplicates(subset=["dataset", "label_cat", "video_id"])
    log.info(f"Built DataFrame: {len(df)} videos")
    return df


# --------------------------------------------------------------------------- #
# Frame loading                                                               #
# --------------------------------------------------------------------------- #

def load_frames_from_paths(frame_paths: List[str], T: int, rng: np.random.Generator):
    """Sample a random contiguous T-window of PIL frames. Returns None if <T frames."""
    n = len(frame_paths)
    if n <= 0:
        raise RuntimeError("Empty frame_paths list")
    if n < T:
        return None
    start = int(rng.integers(0, n - T + 1))
    selected = frame_paths[start : start + T]
    frames = [Image.open(p).convert("RGB") for p in selected]
    return frames, list(range(start, start + T))


# --------------------------------------------------------------------------- #
# Output writers                                                              #
# --------------------------------------------------------------------------- #

def save_embedding_npz(embedding: torch.Tensor, out_dir: Path, video_id: str) -> None:
    """Save [T, D] embedding as compressed .npz via tmp→rename (atomic)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}.npz"
    tmp_path = out_dir / f"{video_id}.tmp.npz"
    np.savez_compressed(tmp_path, embedding=embedding.numpy().astype(np.float32, copy=False))
    tmp_path.replace(out_path)


def write_catalogue(layer_dir: Path, new_rows: List[dict]) -> None:
    """Append rows to <layer_dir>/catalogue.csv, dedup by video_id, atomic swap.

    video_id is forced to string on read/write: pandas' CSV reader infers int
    from all-numeric ids (e.g. FF++ '957'), which breaks sort/dedup when merged
    with new rows whose ids come in as str from JSON keys.
    """
    path = layer_dir / "catalogue.csv"
    if not new_rows and not path.exists():
        return
    new_df = pd.DataFrame(new_rows)
    if not new_df.empty:
        new_df["video_id"] = new_df["video_id"].astype(str)
    if path.exists():
        existing = pd.read_csv(path, dtype={"video_id": str})
        combined = pd.concat([existing, new_df], ignore_index=True) if len(new_df) else existing
        combined = combined.drop_duplicates(subset="video_id", keep="last")
    else:
        combined = new_df
    if not combined.empty:
        combined["video_id"] = combined["video_id"].astype(str)
        combined = combined.sort_values("video_id").reset_index(drop=True)
    tmp = path.with_suffix(".tmp.csv")
    combined.to_csv(tmp, index=False)
    tmp.replace(path)


def write_run_config(
    model_dir: Path,
    cfg: dict,
    layer_dims: Dict[str, Optional[int]],
    completed: bool,
    created_utc: Optional[str] = None,
):
    """Write resolved config + runtime metadata as run_config.json at the model-dir level.

    Called twice per run: once at start (completed=False), once at end
    (completed=True). Pass the original created_utc to preserve it on resume.
    """
    payload = {
        "created_utc":   created_utc or datetime.now(timezone.utc).isoformat(),
        "completed":     completed,
        "completed_utc": datetime.now(timezone.utc).isoformat() if completed else None,
        "layer_dims":    layer_dims,
        **cfg,
    }
    path = model_dir / "run_config.json"
    tmp = path.with_suffix(".tmp.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def load_run_config(model_dir: Path) -> Optional[dict]:
    path = model_dir / "run_config.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Main loop                                                                   #
# --------------------------------------------------------------------------- #

CATALOGUE_FLUSH_EVERY = 50


def _label_name(label: int) -> str:
    return "real" if label == 1 else ("fake" if label == 0 else "unknown")


def run_embedding(cfg: dict, embedder, df: pd.DataFrame, model_dir: Path):
    """
    Loop over videos, run one forward pass per video, save one .npz per layer.

    A video is considered done when ALL expected layer files exist; otherwise
    we re-embed all layers (cheaper than tracking per-layer state). Catalogues
    are flushed every CATALOGUE_FLUSH_EVERY videos so a crash loses at most
    ~50 rows of CSV state (the .npz files on disk remain authoritative).
    """
    required = {"video_id", "label", "label_cat", "frame_paths"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing columns: {missing}")

    T            = cfg["T"]
    batch_size   = cfg.get("batch_size")
    dataset_name = cfg["dataset_name"]
    max_videos   = cfg.get("max_videos")
    rng          = np.random.default_rng(cfg["seed"])

    layer_names: List[str] = embedder.layer_names
    layer_dirs: Dict[str, Path] = {ln: _layer_dir(model_dir, ln) for ln in layer_names}
    layer_dims: Dict[str, Optional[int]] = {ln: embedder.layer_dim(ln) for ln in layer_names}
    catalogue_rows: Dict[str, List[dict]] = {ln: [] for ln in layer_names}

    failures: List[tuple] = []
    times: List[float] = []
    skipped_existing = 0
    df_video_ids = set(df["video_id"].tolist())

    # Orphan scan per layer: .npz files not in input df (informational only).
    for ln, ldir in layer_dirs.items():
        orphans: List[str] = []
        if ldir.exists():
            for npz in ldir.rglob("*.npz"):
                stem = npz.stem
                if stem not in df_video_ids and stem.replace("__", "/") not in df_video_ids:
                    orphans.append(str(npz.relative_to(ldir)))
        if orphans:
            log.warning(
                f"[{ln}]: {len(orphans)} .npz file(s) not in input df "
                f"(will be included in catalogue). First: {orphans[:3]}"
            )

    N = len(df) if max_videos is None else min(len(df), max_videos)

    def _flush_all():
        for ln in layer_names:
            write_catalogue(layer_dirs[ln], catalogue_rows[ln])

    interrupted = False
    try:
        for j, row in enumerate(
            tqdm(df.itertuples(index=False), total=N, desc="Embedding", unit="video"),
            start=1,
        ):
            if max_videos is not None and j > max_videos:
                break

            t0 = time.perf_counter()
            video_id = row.video_id
            frame_paths = list(row.frame_paths)
            label = int(row.label)
            label_cat = row.label_cat

            clean_video_id = video_id.replace("/", "__").replace("\\", "__")
            out_files: Dict[str, Path] = {
                ln: layer_dirs[ln] / dataset_name / label_cat / f"{clean_video_id}.npz"
                for ln in layer_names
            }

            # Skip only if every expected layer file is already on disk.
            if all(p.exists() for p in out_files.values()):
                skipped_existing += 1
                for ln in layer_names:
                    catalogue_rows[ln].append({
                        "dataset":        dataset_name,
                        "video_id":       video_id,
                        "label":          label,
                        "label_name":     _label_name(label),
                        "label_cat":      label_cat,
                        "embedding_file": str(out_files[ln].as_posix()),
                        "n_frames":       int(T),
                        "embedding_dim":  layer_dims[ln],
                        "layer":          ln,
                    })
                if skipped_existing % CATALOGUE_FLUSH_EVERY == 0:
                    _flush_all()
                continue

            source_path = str(Path(frame_paths[0]).parent) if frame_paths else ""

            try:
                result = load_frames_from_paths(frame_paths, T, rng)
                if result is None:
                    failures.append((video_id, source_path, f"Too few frames (<{T})"))
                    log.error(f"[{video_id}] too few frames (<{T}) at {source_path}")
                    continue
                frames, _ = result

                embeddings = embedder.embed_frames(frames, batch_size=batch_size)

                for ln in layer_names:
                    if ln not in embeddings:
                        raise RuntimeError(f"Expected layer '{ln}' missing from embedder output")
                    emb = embeddings[ln]
                    if emb.ndim != 2 or emb.shape[0] != T:
                        raise RuntimeError(
                            f"Bad shape {tuple(emb.shape)} for layer {ln}, expected ({T}, D)"
                        )
                    save_embedding_npz(emb, layer_dirs[ln] / dataset_name / label_cat, clean_video_id)
                    catalogue_rows[ln].append({
                        "dataset":        dataset_name,
                        "video_id":       video_id,
                        "label":          label,
                        "label_name":     _label_name(label),
                        "label_cat":      label_cat,
                        "embedding_file": str(out_files[ln].as_posix()),
                        "n_frames":       int(T),
                        "embedding_dim":  int(emb.shape[1]),
                        "layer":          ln,
                    })

                times.append(time.perf_counter() - t0)
                if len(times) % CATALOGUE_FLUSH_EVERY == 0:
                    _flush_all()

            except Exception as e:
                failures.append((video_id, source_path, str(e)))
                log.error(f"[{video_id}] failed: {e}")

    except KeyboardInterrupt:
        interrupted = True
        log.warning("KeyboardInterrupt — flushing catalogues before exit.")
        raise
    finally:
        _flush_all()
        if interrupted:
            log.info(
                f"Flushed on interrupt. Progress: embedded={len(times)}, "
                f"skipped={skipped_existing}, failures={len(failures)}."
            )

    if not times and skipped_existing > 0:
        log.info(f"All {skipped_existing} videos already embedded — nothing to do.")
    elif not times:
        log.warning(f"0 videos embedded successfully out of {N}.")
    elif skipped_existing > 0:
        log.info(f"Embedded {len(times)} new videos, skipped {skipped_existing} already existing.")
    else:
        log.info(f"Embedded {len(times)} videos.")

    return failures


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _validate_resume(
    prior: Optional[dict],
    cfg: dict,
    resume: bool,
    model_dir: Path,
) -> Optional[str]:
    """Gate fresh-start vs resume. Returns original created_utc or None."""
    if prior is None:
        if resume:
            raise SystemExit(
                f"--resume requested but no run_config.json found in {model_dir}. "
                "Run without -r to start fresh."
            )
        return None

    completed = prior.get("completed", False)
    if completed:
        if resume:
            raise SystemExit(
                f"Nothing to resume: run in {model_dir} is already marked completed."
            )
        raise SystemExit(
            f"Output dir {model_dir} already has a completed run. "
            "Change output_base_dir or delete the directory to start over."
        )

    if not resume:
        raise SystemExit(
            f"Incomplete run detected in {model_dir} (completed=false). "
            "Re-run with -r / --resume to continue, or delete the directory to start over."
        )

    # Resume — validate config compatibility.
    mismatches = []
    if prior.get("T") != cfg["T"]:
        mismatches.append(f"T: prior={prior.get('T')}, current={cfg['T']}")
    prior_model = (prior.get("model") or {}).get("name")
    if prior_model != cfg["model"]["name"]:
        mismatches.append(f"model.name: prior={prior_model}, current={cfg['model']['name']}")
    prior_ds = prior.get("dataset_name")
    if prior_ds != cfg["dataset_name"]:
        mismatches.append(f"dataset_name: prior={prior_ds}, current={cfg['dataset_name']}")

    # Extract spec must match so layer dirs stay in sync with run_config.
    prior_ex = prior.get("extract") or {}
    cur_ex = cfg.get("extract") or {}
    prior_blocks = sorted(set(prior_ex.get("blocks") or []))
    cur_blocks = sorted(set(cur_ex.get("blocks") or []))
    if prior_blocks != cur_blocks:
        mismatches.append(f"extract.blocks: prior={prior_blocks}, current={cur_blocks}")
    if bool(prior_ex.get("include_final", True)) != bool(cur_ex.get("include_final", True)):
        mismatches.append(
            f"extract.include_final: prior={prior_ex.get('include_final', True)}, "
            f"current={cur_ex.get('include_final', True)}"
        )

    if mismatches:
        raise SystemExit(
            "Cannot resume — current config conflicts with prior run:\n  "
            + "\n  ".join(mismatches)
        )

    return prior.get("created_utc")


def main():
    ap = argparse.ArgumentParser(description="Run CLIP embedding from a YAML config.")
    ap.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    ap.add_argument("-r", "--resume", action="store_true",
                    help="Resume an incomplete prior run in the same output dir.")
    ap.add_argument("--max_videos", type=int, default=None, help="Override config max_videos.")
    ap.add_argument("--device", type=str, default=None, help="Override config device.")
    args = ap.parse_args()

    cfg = _load_config(args.config)

    if args.max_videos is not None:
        cfg["max_videos"] = args.max_videos
    if args.device is not None:
        cfg["device"] = args.device

    json_path = _resolve_path(cfg["input_json"])
    base_dir  = _resolve_path(cfg["output_base_dir"])
    blocks, include_final = _parse_extract(cfg)

    # Resolve output dir & set up logger before any heavy work so the log file
    # captures the full run (including model load). FileHandler appends, so
    # resumed runs accumulate into the same embed.log.
    model_dir = _get_model_dir(base_dir, cfg["model"]["name"])
    create_logger(str(model_dir / "embed.log"))
    # Stop propagation to the root logger — some deps (open_clip/huggingface_hub)
    # attach a basicConfig handler that duplicates every line unformatted.
    log.propagate = False
    log.info("=" * 60)
    log.info(f"Starting run | config={args.config} | resume={args.resume}")
    log.info(f"Dataset: {cfg['dataset_name']}  T={cfg['T']}  seed={cfg['seed']}  "
             f"max_videos={cfg.get('max_videos')}  device={cfg['device']}")
    log.info(f"Model: {cfg['model']['name']} (pretrained={cfg['model']['pretrained']})")
    log.info(f"Extract: blocks={blocks}, include_final={include_final}")
    log.info(f"Output dir: {model_dir}")

    df = build_df_from_repo_json(json_path, cfg["dataset_name"])

    embedder = CLIP_EMBEDDER(
        model_name=cfg["model"]["name"],
        pretrained=cfg["model"]["pretrained"],
        device=cfg["device"],
        blocks=blocks,
        include_final=include_final,
    )

    layer_dims = {ln: embedder.layer_dim(ln) for ln in embedder.layer_names}
    log.info(f"Layers: {embedder.layer_names}  dims: {layer_dims}")

    prior = load_run_config(model_dir)
    created_utc = _validate_resume(prior, cfg, args.resume, model_dir)
    if created_utc:
        log.info(f"Resuming run originally started {created_utc}.")

    # Mark in-progress up front so a crash leaves completed=false on disk.
    write_run_config(model_dir, cfg, layer_dims, completed=False, created_utc=created_utc)

    failures = run_embedding(cfg, embedder, df, model_dir)

    write_run_config(model_dir, cfg, layer_dims, completed=True, created_utc=created_utc)

    log.info(f"Embedding finished. Failures: {len(failures)}")
    if failures:
        log.info(f"First failure: {failures[0]}")


if __name__ == "__main__":
    main()
