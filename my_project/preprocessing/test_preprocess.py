"""
Visual test for FaceAligner preprocessing.

For each sampled video this script:
  1. Extracts T frames with OpenCV
  2. Runs FaceAligner.preprocess() to obtain aligned crops
  3. Saves a PNG contact sheet with two rows:
       Row 1 (Original):  sampled frames at PREVIEW_STRIDE spacing
       Row 2 (Aligned):   the corresponding aligned 336×336 crops

The contact sheets let you verify:
  - Face is detected and cropped correctly
  - The crop is identical across all frames (temporal consistency)
  - Both real and fake videos are handled

Usage (from project root):
    python preprocessing/test_preprocess.py
    python preprocessing/test_preprocess.py --n-real 3 --n-fake 3 --device cuda
    python preprocessing/test_preprocess.py --index /path/to/index.parquet --out-dir /tmp/out
"""

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.preprocess import FaceAligner


# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_INDEX   = "artifacts/celeb_df_index.parquet"
DEFAULT_OUT_DIR = "preprocessing/test_output"
T               = 96    # frames to extract per video
PREVIEW_STRIDE  = 16    # show every Nth frame in the grid (96 frames → 6 columns)
N_REAL          = 2     # real videos to sample
N_FAKE          = 2     # fake videos to sample
THUMB_W         = 200   # thumbnail width in the grid (px)
ROW_LABEL_H     = 18    # height of the row-label bar
COL_LABEL_H     = 14    # height of the frame-index bar above each column
# ─────────────────────────────────────────────────────────────────────────────


def extract_frames(video_path: str, n: int) -> list[Image.Image] | None:
    """Extract the first n frames from a video as RGB PIL Images."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    frames = []
    for _ in range(min(n, total)):
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))

    cap.release()
    return frames or None


def make_comparison_grid(
    orig_frames: list[Image.Image],
    aligned_frames: list[Image.Image],
) -> Image.Image:
    """
    Build a two-row contact sheet.

    Row 1 — Original frames:  letterboxed to THUMB_W wide, natural height
    Row 2 — Aligned frames:   square THUMB_W × THUMB_W crops

    Each column shows the same frame index; a label bar sits above each row.
    """
    indices = list(range(0, len(orig_frames), PREVIEW_STRIDE))
    n_cols  = len(indices)

    def _thumb_orig(img: Image.Image) -> Image.Image:
        ar = img.height / img.width
        return img.resize((THUMB_W, int(THUMB_W * ar)), Image.LANCZOS)

    orig_thumbs    = [_thumb_orig(orig_frames[i]) for i in indices]
    aligned_thumbs = [aligned_frames[i].resize((THUMB_W, THUMB_W), Image.LANCZOS)
                      for i in indices]

    row_h_orig    = max(t.height for t in orig_thumbs)
    row_h_aligned = THUMB_W

    total_w = n_cols * THUMB_W
    total_h = (COL_LABEL_H
               + ROW_LABEL_H + row_h_orig
               + ROW_LABEL_H + row_h_aligned)

    canvas = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    draw   = ImageDraw.Draw(canvas)

    # Row labels (left edge, in the label bar for that row)
    y_orig_label    = COL_LABEL_H
    y_aligned_label = COL_LABEL_H + ROW_LABEL_H + row_h_orig
    draw.text((4, y_orig_label + 2),    "Original", fill=(210, 210, 210))
    draw.text((4, y_aligned_label + 2), "Aligned",  fill=(140, 220, 140))

    for col, (frame_idx, orig_t, aligned_t) in enumerate(
        zip(indices, orig_thumbs, aligned_thumbs)
    ):
        x = col * THUMB_W

        # Frame-index label at the very top
        draw.text((x + 4, 2), f"f{frame_idx}", fill=(180, 180, 180))

        # Original row
        y_orig = COL_LABEL_H + ROW_LABEL_H + (row_h_orig - orig_t.height) // 2
        canvas.paste(orig_t, (x, y_orig))

        # Aligned row
        y_aligned = COL_LABEL_H + ROW_LABEL_H + row_h_orig + ROW_LABEL_H
        canvas.paste(aligned_t, (x, y_aligned))

    return canvas


def run_test(
    index_path: str,
    out_dir: str,
    n_real: int,
    n_fake: int,
    device: str,
    face_scale: float,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading index: {index_path}")
    df = pd.read_parquet(index_path)
    print(f"Index loaded: {len(df)} videos")

    real_df = df[df["label"] == 1]
    fake_df = df[df["label"] == 0]
    sample  = pd.concat([
        real_df.sample(n=min(n_real, len(real_df)), random_state=42),
        fake_df.sample(n=min(n_fake, len(fake_df)), random_state=42),
    ]).reset_index(drop=True)

    print(f"Testing {len(sample)} videos  "
          f"({min(n_real, len(real_df))} real, {min(n_fake, len(fake_df))} fake)\n")

    print(f"face_scale={face_scale}  (1.0=tight ArcFace, lower=more context)\n")
    aligner = FaceAligner(output_size=336, face_scale=face_scale, device=device)
    ok_count = fail_count = 0

    for _, row in sample.iterrows():
        video_id   = str(row["video_id"])
        video_path = str(row["video_path"])
        label_name = "real" if row["label"] == 1 else "fake"

        print(f"[{label_name}] {video_id}")

        # 1. Extract frames
        orig_frames = extract_frames(video_path, T)
        if orig_frames is None:
            print(f"  SKIP — could not extract frames from {video_path}\n")
            fail_count += 1
            continue
        print(f"  extracted {len(orig_frames)} frames  ({orig_frames[0].size[0]}x{orig_frames[0].size[1]})")

        # 2. Face alignment
        aligned_frames = aligner.preprocess(orig_frames)
        if aligned_frames is None:
            print("  SKIP — no face detected\n")
            fail_count += 1
            continue
        print(f"  aligned   {len(aligned_frames)} frames  ({aligned_frames[0].size[0]}x{aligned_frames[0].size[1]})")

        # 3. Save contact sheet
        grid      = make_comparison_grid(orig_frames, aligned_frames)
        safe_id   = video_id.replace("/", "__").replace("\\", "__")
        save_path = out_path / f"{label_name}_{safe_id}.png"
        grid.save(save_path)
        print(f"  saved  → {save_path}\n")
        ok_count += 1

    print(f"Done — {ok_count} saved, {fail_count} skipped")
    print(f"Output directory: {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visual test for FaceAligner: saves original vs aligned contact sheets."
    )
    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX,
        help=f"Path to dataset index parquet (default: {DEFAULT_INDEX})",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Directory for output PNG files (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--n-real", type=int, default=N_REAL,
        help=f"Real videos to sample (default: {N_REAL})",
    )
    parser.add_argument(
        "--n-fake", type=int, default=N_FAKE,
        help=f"Fake videos to sample (default: {N_FAKE})",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for MTCNN detection (default: cpu)",
    )
    parser.add_argument(
        "--face-scale", type=float, default=0.75,
        help="Fraction of output canvas occupied by the canonical face region. "
             "Lower = more zoom-out / padding (default: 0.75)",
    )
    args = parser.parse_args()

    run_test(
        index_path=args.index,
        out_dir=args.out_dir,
        n_real=args.n_real,
        n_fake=args.n_fake,
        device=args.device,
        face_scale=args.face_scale,
    )