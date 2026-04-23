"""Preprocess WildDeepfakes: extract 96 contiguous frames per internal video.

WildDeepfakes is distributed as tar archives (despite the .tar.gz extension,
they are uncompressed POSIX tar — tarfile.open auto-detects). Each tar holds
multiple videos:
    ./{tar_id}/{fake|real}/{video_id}/{frame_num}.png

Frames are already face-cropped 224x224 PNGs, so no face detection is run.
For each internal video we:
  1. sort PNG members by numeric frame id
  2. pick a random contiguous 96-frame window
  3. copy the raw PNG bytes (no decode/resize) to
     {sub_dataset}/frames/{tar_id}_{video_id}/NNN.png

Resumable: skips videos whose output dir already has exactly num_frames PNGs.
"""

import concurrent.futures
import datetime
import logging
import os
import random
import sys
import tarfile
import time
from pathlib import Path

import yaml
from tqdm import tqdm


SUB_DATASETS = ['fake_test', 'fake_train', 'real_test', 'real_train']


def create_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def process_tar(tar_path, output_root, num_frames, logger):
    """Extract `num_frames` contiguous PNGs per internal video in this tar."""
    tar_id = tar_path.name.split('.')[0]  # "1.tar.gz" -> "1"

    try:
        with tarfile.open(tar_path, 'r') as tar:
            # Group PNG members by video_id within this tar
            videos = {}  # video_id -> list[(frame_num, member)]
            for member in tar.getmembers():
                if not member.isfile() or not member.name.endswith('.png'):
                    continue
                parts = member.name.strip('./').split('/')
                if len(parts) != 4:
                    continue
                _, _label, video_id, fname = parts
                try:
                    frame_num = int(fname.rsplit('.', 1)[0])
                except ValueError:
                    continue
                videos.setdefault(video_id, []).append((frame_num, member))

            for video_id, frames in videos.items():
                frames.sort(key=lambda x: x[0])
                if len(frames) < num_frames:
                    logger.warning(
                        f"{tar_path.name}: video {video_id} has {len(frames)} frames "
                        f"(< {num_frames}), skipping"
                    )
                    continue

                out_dir = output_root / f"{tar_id}_{video_id}"
                if out_dir.exists():
                    existing = list(out_dir.glob('*.png'))
                    if len(existing) == num_frames:
                        continue
                    # Stale output from a prior run with a different num_frames.
                    # Clear so we don't leave orphaned NNN.png from the old window.
                    for p in existing:
                        p.unlink()

                start = random.randint(0, len(frames) - num_frames)
                selected = frames[start:start + num_frames]

                out_dir.mkdir(parents=True, exist_ok=True)
                for idx, (_, member) in enumerate(selected):
                    src = tar.extractfile(member)
                    if src is None:
                        continue
                    (out_dir / f"{idx:03d}.png").write_bytes(src.read())
    except Exception as e:
        logger.error(f"Failed processing {tar_path.name}: {e}")


def preprocess_subdataset(sub_path, num_frames, logger):
    tar_paths = sorted(sub_path.glob('*.tar.gz'))
    if not tar_paths:
        logger.error(f"No tars found in {sub_path}")
        return

    logger.info(f"{len(tar_paths)} tars found in {sub_path}")
    output_root = sub_path / 'frames'
    output_root.mkdir(exist_ok=True)

    start_time = time.monotonic()
    num_workers = min(os.cpu_count() or 4, 8)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(process_tar, tp, output_root, num_frames, logger)
                   for tp in tar_paths]
        for fut in tqdm(concurrent.futures.as_completed(futures),
                        total=len(futures), desc=sub_path.name):
            fut.result()  # surface unexpected exceptions (process_tar logs + swallows its own)

    mins = (time.monotonic() - start_time) / 60
    logger.info(f"{sub_path.name}: done in {mins:.2f} min")


if __name__ == '__main__':
    yaml_path = './config.yaml'
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    cfg = config['preprocess_wildd']
    dataset_name = 'WildDeepfakes'
    dataset_root_path = cfg['dataset_root_path']['default']
    num_frames = cfg['num_frames']['default']

    dataset_path = Path(dataset_root_path) / dataset_name
    if not dataset_path.exists():
        print(f"ERROR: dataset path not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs('./logs', exist_ok=True)
    logger = create_logger(f'./logs/{dataset_name}.log')
    logger.info(f"=== {dataset_name} preprocess @ {datetime.datetime.now().isoformat(timespec='seconds')} ===")
    logger.info(f"dataset_path={dataset_path}  num_frames={num_frames}")

    total_start = time.monotonic()
    for sub in SUB_DATASETS:
        sub_path = dataset_path / sub
        if not sub_path.exists():
            logger.error(f"Missing sub-dataset: {sub_path}")
            sys.exit(1)
        preprocess_subdataset(sub_path, num_frames, logger)

    logger.info(f"All splits complete in {(time.monotonic() - total_start) / 60:.2f} min")
