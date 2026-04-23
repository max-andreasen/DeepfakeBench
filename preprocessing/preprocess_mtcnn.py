# MTCNN-based preprocessing for deepfake datasets.
# Detects faces across all frames, computes a per-video union bounding box,
# and crops all frames to that same region for temporal consistency.

import os
import sys
import json
import time
import cv2
import yaml
import logging
import datetime
import glob
import random
import shutil
import concurrent.futures
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image

import torch
from facenet_pytorch import MTCNN


def create_logger(log_path):
    logger = logging.getLogger('mtcnn_preprocess')
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def interpolate_bboxes(per_frame_bboxes, total_frames):
    """
    Fill in missing bounding boxes by linearly interpolating from the nearest
    detected frames on either side.

    Args:
        per_frame_bboxes: dict mapping frame_idx (int) -> {bbox, confidence}
                          Only contains frames where MTCNN succeeded.
        total_frames: Total number of frames in the video.

    Returns:
        full_bboxes: dict mapping frame_idx (int) -> {bbox, confidence, interpolated}
                     for ALL frames 0..total_frames-1.
    """
    detected_indices = sorted(per_frame_bboxes.keys())
    if len(detected_indices) == 0:
        return {}

    full_bboxes = {}

    # Copy detected frames
    for idx in detected_indices:
        full_bboxes[idx] = {
            'bbox': per_frame_bboxes[idx]['bbox'],
            'confidence': per_frame_bboxes[idx]['confidence'],
            'interpolated': False,
        }

    # Interpolate missing frames
    for idx in range(total_frames):
        if idx in full_bboxes:
            continue

        # Find nearest detected frames on either side
        prev_idx = None
        next_idx = None
        for di in detected_indices:
            if di < idx:
                prev_idx = di
            elif di > idx:
                next_idx = di
                break

        if prev_idx is not None and next_idx is not None:
            # Linear interpolation
            t = (idx - prev_idx) / (next_idx - prev_idx)
            prev_bbox = np.array(per_frame_bboxes[prev_idx]['bbox'])
            next_bbox = np.array(per_frame_bboxes[next_idx]['bbox'])
            interp_bbox = (prev_bbox * (1 - t) + next_bbox * t).tolist()
            confidence = 0.0
        elif prev_idx is not None:
            interp_bbox = per_frame_bboxes[prev_idx]['bbox']
            confidence = 0.0
        else:
            interp_bbox = per_frame_bboxes[next_idx]['bbox']
            confidence = 0.0

        full_bboxes[idx] = {
            'bbox': interp_bbox,
            'confidence': confidence,
            'interpolated': True,
        }

    return full_bboxes


def compute_union_bbox(all_bboxes, frame_width, frame_height, padding_scale=1.3):
    """
    Compute the union bounding box across all frames, then expand by padding_scale
    and make it square. Clamp to frame dimensions.

    Args:
        all_bboxes: dict mapping frame_idx -> {bbox: [x1,y1,x2,y2], ...}
        frame_width: Width of the original frames.
        frame_height: Height of the original frames.
        padding_scale: Scale factor to expand the union bbox.

    Returns:
        union_bbox: [x1, y1, x2, y2] raw union
        padded_bbox: [x1, y1, x2, y2] after padding + square + clamp
    """
    bboxes = np.array([v['bbox'] for v in all_bboxes.values()])
    x1 = bboxes[:, 0].min()
    y1 = bboxes[:, 1].min()
    x2 = bboxes[:, 2].max()
    y2 = bboxes[:, 3].max()
    union_bbox = [float(x1), float(y1), float(x2), float(y2)]

    # Expand by padding_scale around center, make square
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half_size = max(x2 - x1, y2 - y1) * padding_scale / 2.0

    px1 = max(0, int(cx - half_size))
    py1 = max(0, int(cy - half_size))
    px2 = min(frame_width, int(cx + half_size))
    py2 = min(frame_height, int(cy + half_size))

    padded_bbox = [px1, py1, px2, py2]
    return union_bbox, padded_bbox


def process_single_video(
    video_path,
    save_path,
    detector,
    output_resolution,
    bbox_padding_scale,
    num_frames,
    logger,
):
    """
    Process a single video: sample a random consecutive clip of num_frames,
    detect faces with MTCNN, compute union bounding box, crop all frames
    to that box, and save results.
    """
    video_stem = video_path.stem

    frames_dir = save_path / 'mtcnn_frames' / video_stem
    bbox_file = save_path / 'mtcnn_bboxes' / f'{video_stem}.json'

    # Skip if already fully processed (exactly num_frames PNGs)
    if frames_dir.exists() and bbox_file.exists():
        existing = len(list(frames_dir.glob('*.png')))
        if existing == num_frames:
            return
        # Incomplete — wipe and redo with a fresh random sample
        logger.info(f"Re-processing {video_path.name} — incomplete ({existing}/{num_frames} frames)")
        shutil.rmtree(frames_dir, ignore_errors=True)
        bbox_file.unlink(missing_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open {video_path}")
        return

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_video_frames < num_frames:
        logger.error(f"Video {video_path.name} has only {total_video_frames} frames (need {num_frames})")
        cap.release()
        return

    # --- Phase 1: Sample random start and read consecutive frames ---
    start_frame = random.randint(0, total_video_frames - num_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    raw_frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()

    if len(raw_frames) < num_frames:
        logger.error(f"Only read {len(raw_frames)}/{num_frames} frames from {video_path.name} (start={start_frame})")
        return

    # --- Phase 2: Batch MTCNN detection ---
    batch_size = 32
    per_frame_bboxes = {}
    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in raw_frames]

    for batch_start in range(0, num_frames, batch_size):
        batch = pil_frames[batch_start:batch_start + batch_size]
        batch_boxes, batch_probs = detector.detect(batch)

        for i, (boxes, probs) in enumerate(zip(batch_boxes, batch_probs)):
            idx = batch_start + i
            if boxes is not None and len(boxes) > 0:
                best = int(np.argmax(probs))
                per_frame_bboxes[idx] = {
                    'bbox': boxes[best].tolist(),
                    'confidence': float(probs[best]),
                }

    # Check detection rate
    detected_count = len(per_frame_bboxes)
    missed_count = num_frames - detected_count
    if detected_count == 0:
        logger.error(f"No faces detected in any frame of {video_path.name}")
        return

    if missed_count > 0 and missed_count / num_frames > 0.1:
        logger.warning(
            f"{video_path.name}: {missed_count}/{num_frames} frames missed "
            f"({missed_count / num_frames:.1%})"
        )

    # --- Phase 3: Interpolate missing bboxes ---
    all_bboxes = interpolate_bboxes(per_frame_bboxes, num_frames)

    # --- Phase 4: Compute union bounding box ---
    union_bbox, padded_bbox = compute_union_bbox(
        all_bboxes, frame_width, frame_height, bbox_padding_scale
    )
    px1, py1, px2, py2 = padded_bbox

    # --- Phase 5: Crop and save frames ---
    frames_dir.mkdir(parents=True, exist_ok=True)
    bbox_dir = save_path / 'mtcnn_bboxes'
    bbox_dir.mkdir(parents=True, exist_ok=True)

    def save_frame(idx, frame):
        cropped = frame[py1:py2, px1:px2]
        if cropped.size == 0:
            return
        resized = cv2.resize(cropped, (output_resolution, output_resolution), interpolation=cv2.INTER_LINEAR)
        out_path = frames_dir / f'{idx:04d}.png'
        cv2.imwrite(str(out_path), resized)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as io_pool:
        futures = []
        for idx, frame in enumerate(raw_frames):
            futures.append(io_pool.submit(save_frame, idx, frame))
        for f in futures:
            f.result()

    # --- Phase 6: Save metadata ---
    per_frame_meta = {}
    for idx in range(num_frames):
        info = all_bboxes[idx]
        per_frame_meta[f'{idx:04d}'] = {
            'bbox': [round(v, 2) for v in info['bbox']],
            'confidence': round(info['confidence'], 4),
            'interpolated': info['interpolated'],
        }

    metadata = {
        'video': video_stem,
        'total_video_frames': total_video_frames,
        'sampled_frames': num_frames,
        'start_frame': start_frame,
        'frame_height': frame_height,
        'frame_width': frame_width,
        'union_bbox': [round(v, 2) for v in union_bbox],
        'union_bbox_padded': padded_bbox,
        'padding_scale': bbox_padding_scale,
        'output_resolution': output_resolution,
        'per_frame': per_frame_meta,
        'missed_frame_count': missed_count,
        'detector': 'mtcnn',
    }

    with open(bbox_file, 'w') as f:
        json.dump(metadata, f, indent=2)



def preprocess_subdataset(sub_dataset_path, output_base_path, detector, output_resolution, bbox_padding_scale, num_frames, logger):
    """Process all videos in a single sub-dataset directory.

    Args:
        sub_dataset_path: Path to source videos (e.g. .../original_sequences/youtube/c23)
        output_base_path: Where to save mtcnn_frames/ and mtcnn_bboxes/.
                          Same as sub_dataset_path if no separate output disk is configured.
    """
    movies_path_list = sorted([
        Path(p) for p in glob.glob(os.path.join(sub_dataset_path, '**/*.mp4'), recursive=True)
    ])
    if len(movies_path_list) == 0:
        logger.error(f"No videos found in {sub_dataset_path}")
        return
    logger.info(f"{len(movies_path_list)} videos found in {sub_dataset_path}")
    logger.info(f"Output directory: {output_base_path}")

    start_time = time.monotonic()

    for movie_path in tqdm(movies_path_list, desc=str(sub_dataset_path)):
        try:
            process_single_video(
                movie_path,
                output_base_path,
                detector,
                output_resolution,
                bbox_padding_scale,
                num_frames,
                logger,
            )
        except Exception as e:
            logger.error(f"Error processing video {movie_path}: {e}")

    duration_minutes = (time.monotonic() - start_time) / 60
    logger.info(f"Sub-dataset {sub_dataset_path} done in {duration_minutes:.2f} minutes")


if __name__ == '__main__':
    # Load config
    yaml_path = './config.yaml'
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    cfg = config['preprocess_mtcnn']
    dataset_name = cfg['dataset_name']['default']
    dataset_root_path = cfg['dataset_root_path']['default']
    comp = cfg['comp']['default']
    device = cfg['device']['default']
    output_resolution = cfg['output_resolution']['default']
    bbox_padding_scale = cfg['bbox_padding_scale']['default']
    num_frames = cfg['num_frames']['default']

    dataset_path = Path(os.path.join(dataset_root_path, dataset_name))

    # Create logger
    log_path = f'./logs/{dataset_name}_mtcnn.log'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = create_logger(log_path)
    logger.info(f"Starting MTCNN preprocessing for {dataset_name}")
    logger.info(f"Config: device={device}, resolution={output_resolution}, padding={bbox_padding_scale}, num_frames={num_frames}")

    # Initialize MTCNN detector (single instance, GPU)
    detector = MTCNN(
        select_largest=True,
        keep_all=False,
        post_process=False,
        device=device,
    )
    logger.info(f"MTCNN detector initialized on {device}")

    # Build sub-dataset paths (FaceForensics++ only for now)
    if dataset_name == 'FaceForensics++':
        sub_dataset_names = [
            "original_sequences/youtube",
            "original_sequences/actors",
            "manipulated_sequences/Deepfakes",
            "manipulated_sequences/Face2Face",
            "manipulated_sequences/FaceSwap",
            "manipulated_sequences/NeuralTextures",
            "manipulated_sequences/FaceShifter",
            "manipulated_sequences/DeepFakeDetection",
        ]
        sub_dataset_paths = [Path(os.path.join(dataset_path, name, comp)) for name in sub_dataset_names]
    elif dataset_name in ('Celeb-DF-v1', 'Celeb-DF-v2'):
        sub_dataset_names = ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]
    elif dataset_name == 'Celeb-DF-v3':
        sub_dataset_paths = [Path(os.path.join(dataset_path, 'Celeb-real')),
                             Path(os.path.join(dataset_path, 'YouTube-real'))]
        for category in ['FaceReenact', 'FaceSwap', 'TalkingFace']:
            cat_dir = Path(os.path.join(dataset_path, 'Celeb-synthesis', category))
            sub_dataset_paths += sorted(p for p in cat_dir.iterdir() if p.is_dir())
    else:
        logger.error(f"Dataset {dataset_name} not yet supported for MTCNN preprocessing")
        sys.exit(1)

    # Validate paths
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    for sub_path in sub_dataset_paths:
        if not sub_path.exists():
            logger.error(f"Sub-dataset path does not exist: {sub_path}")
            sys.exit(1)

    # Process each sub-dataset
    total_start = time.monotonic()
    for sub_dataset_path in sub_dataset_paths:
        logger.info(f"Processing: {sub_dataset_path}")
        preprocess_subdataset(
            sub_dataset_path, sub_dataset_path, detector, output_resolution, bbox_padding_scale, num_frames, logger
        )

    total_minutes = (time.monotonic() - total_start) / 60
    logger.info(f"All done! Total time: {total_minutes:.2f} minutes")
