"""
Face alignment and cropping for deepfake detection.

Temporal consistency guarantee:
    Face is detected in a single reference frame (middle frame by default).
    The resulting 2x3 similarity transform is applied identically to all T
    frames, so every frame in the video has the same crop and alignment.
    This is required for the temporal transformer, which must see the same
    face region across the 32-frame (or 3x32-frame) windows.

Standard approach following deepfake detection literature (FF++, DFDC):
    1. MTCNN face detection with 5-point landmarks
    2. Similarity transform mapping detected landmarks → canonical ArcFace
       reference positions (originally 112x112, scaled to output_size)
    3. Fixed-size square crop at output_size x output_size
       (336 for CLIP ViT-L-14-336)
    4. Fallback: bounding-box crop with margin if landmark estimation fails

Usage:
    aligner = FaceAligner(output_size=336, device="cuda")
    aligned_frames = aligner.preprocess(frames)   # frames: list[PIL.Image]
    if aligned_frames is None:
        ...  # no face found — discard video

Integration with CLIP_embedder:
    In CLIP_embedder._run_embedding_loop, call aligner.preprocess(frames)
    on the output of _extract_frames() before passing to _embed_batch().
"""

import numpy as np
import cv2
from PIL import Image

from facenet_pytorch import MTCNN


# ---------------------------------------------------------------------------
# ArcFace 5-point reference landmarks at 112 × 112
# (left eye, right eye, nose, left mouth corner, right mouth corner)
# ---------------------------------------------------------------------------
_ARCFACE_112 = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # left mouth
        [70.7299, 92.2041],  # right mouth
    ],
    dtype=np.float32,
)


class FaceAligner:
    """
    Temporally consistent face alignment for deepfake detection.

    Detects a face once in a reference frame and applies the same affine
    transform to every frame in the video. This ensures all T frames share
    an identical crop regardless of subject motion.
    """

    def __init__(
        self,
        output_size: int = 336,
        face_scale: float = 0.75,
        bbox_margin: float = 0.35,
        device: str = "cpu",
    ):
        """
        Args:
            output_size: Edge length (px) of the square output crop.
                         336 matches CLIP ViT-L-14-336 native resolution.
            face_scale:  Fraction of output_size that the canonical face region
                         occupies. The remainder becomes uniform padding on all
                         four sides. Smaller values zoom out and show more
                         context (forehead, chin, cheeks).
                           1.0  → tight ArcFace crop, face fills the whole canvas
                           0.75 → face in central 252px of 336px, 42px margin
                           0.65 → face in central 218px of 336px, 59px margin
                         Typical deepfake detection pipelines use 0.65–0.80.
            bbox_margin: Fractional margin added around the bounding box when
                         falling back to a box-only crop (no landmarks).
            device:      Torch device string for MTCNN ("cpu" or "cuda").
        """
        self.output_size = output_size
        self.face_scale  = face_scale
        self.bbox_margin = bbox_margin

        # Place the canonical landmarks inside a face_scale-sized sub-region
        # centred in the output canvas, so that face_scale < 1 adds padding.
        face_region = output_size * face_scale
        offset      = (output_size - face_region) / 2.0
        self._ref_lmks = _ARCFACE_112 * (face_region / 112.0) + offset

        # MTCNN: select the largest detected face, no internal crop
        self._detector = MTCNN(
            select_largest=True,
            keep_all=False,
            post_process=False,  # we handle warping ourselves
            device=device,
        )

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect(
        self, frame_rgb: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Run MTCNN on a single HxWx3 uint8 RGB frame.

        Returns:
            (box [4], landmarks [5,2]) on success,
            (box [4], None)            if only bounding box was found,
            (None,    None)            if no face detected.
        """
        pil = Image.fromarray(frame_rgb)
        boxes, probs, landmarks = self._detector.detect(pil, landmarks=True)

        if boxes is None or len(boxes) == 0:
            return None, None

        best = int(np.argmax(probs))
        box = boxes[best]                                   # [x1, y1, x2, y2]
        lmks = landmarks[best] if landmarks is not None else None  # [5, 2]
        return box, lmks

    def _find_reference_detection(
        self, frames: list[np.ndarray]
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Detect a face in the middle frame, spiralling outward as fallback.

        Up to 15 candidate frames are tried so that brief occlusions or
        blurry frames near the centre do not cause the whole video to fail.

        Returns (box, landmarks) or (box, None) or (None, None).
        """
        n = len(frames)
        mid = n // 2

        candidates: list[int] = [mid]
        for offset in range(1, n):
            for idx in (mid - offset, mid + offset):
                if 0 <= idx < n and idx not in candidates:
                    candidates.append(idx)
            if len(candidates) >= 15:
                break

        for idx in candidates:
            box, lmks = self._detect(frames[idx])
            if box is not None:
                return box, lmks

        return None, None

    # ------------------------------------------------------------------
    # Transform computation
    # ------------------------------------------------------------------

    def _transform_from_landmarks(
        self, lmks: np.ndarray
    ) -> np.ndarray | None:
        """
        Compute a 2x3 similarity transform (uniform scale + rotation +
        translation) that maps the 5 detected MTCNN landmarks to the
        canonical ArcFace reference positions at self.output_size.

        Uses LMEDS for robustness; returns None if estimation fails.
        """
        M, _ = cv2.estimateAffinePartial2D(
            lmks.astype(np.float32),
            self._ref_lmks,
            method=cv2.LMEDS,
        )
        return M  # 2×3 float64, or None

    def _transform_from_box(
        self, box: np.ndarray, frame_hw: tuple[int, int]
    ) -> np.ndarray:
        """
        Compute a 23 affine matrix that crops the padded bounding box and
        scales it to output_size x output_size.

        Used as fallback when landmark estimation is unavailable or fails.
        """
        h, w = frame_hw
        x1, y1, x2, y2 = box

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        half = max(x2 - x1, y2 - y1) * (1.0 + self.bbox_margin) / 2.0

        # Clamp origin to frame boundaries
        x1c = max(0.0, cx - half)
        y1c = max(0.0, cy - half)
        crop_size = min(w - x1c, h - y1c, 2.0 * half)

        scale = self.output_size / crop_size
        M = np.array(
            [
                [scale, 0.0, -x1c * scale],
                [0.0, scale, -y1c * scale],
            ],
            dtype=np.float64,
        )
        return M

    # ------------------------------------------------------------------
    # Warping
    # ------------------------------------------------------------------

    def _warp(self, frame: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Apply a 2×3 affine matrix; return an output_size × output_size crop."""
        return cv2.warpAffine(
            frame,
            M,
            (self.output_size, self.output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(
        self,
        frames: "list[Image.Image] | np.ndarray",
    ) -> "list[Image.Image] | None":
        """
        Align and crop faces across an entire video sequence.

        A single face detection is performed on the reference frame. The
        derived 2x3 affine matrix is then applied to every frame, guaranteeing
        pixel-level crop consistency across the full T-frame sequence. This
        means 32-frame training windows and the three 3x32-frame inference
        windows all come from the same aligned coordinate space.

        Args:
            frames: Either a list of T PIL Images (RGB) or a numpy array of
                    shape (T, H, W, 3) in uint8 RGB format.

        Returns:
            List of T PIL Images of size (output_size x output_size), or
            None if no face could be detected in any candidate frame
            (the caller should discard this video).
        """
        # Normalise to a list of uint8 RGB numpy arrays
        if isinstance(frames, np.ndarray):
            if frames.ndim != 4 or frames.shape[3] != 3:
                raise ValueError(
                    f"Expected array of shape (T, H, W, 3), got {frames.shape}"
                )
            np_frames = [frames[i] for i in range(frames.shape[0])]
        else:
            np_frames = [np.asarray(f) for f in frames]

        # 1. Detect in reference frame (middle ± fallbacks)
        box, lmks = self._find_reference_detection(np_frames)
        if box is None:
            return None

        # 2. Choose the best available transform
        M = None
        if lmks is not None:
            M = self._transform_from_landmarks(lmks)

        if M is None:
            # Landmark estimation failed or unavailable — use padded bbox crop
            h, w = np_frames[0].shape[:2]
            M = self._transform_from_box(box, (h, w))

        # 3. Apply the SAME transform to every frame
        aligned = [Image.fromarray(self._warp(f, M)) for f in np_frames]
        return aligned
