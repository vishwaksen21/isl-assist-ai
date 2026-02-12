from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


NUM_LANDMARKS = 21
DIMS = 3
FEATURES = NUM_LANDMARKS * DIMS  # 63


@dataclass(frozen=True)
class NormalizationConfig:
    eps: float = 1e-6


def flip_landmarks_x_raw01(landmarks_21x3: np.ndarray) -> np.ndarray:
    """Flip x in MediaPipe normalized coordinate space (x in [0,1])."""
    flipped = landmarks_21x3.copy()
    flipped[:, 0] = 1.0 - flipped[:, 0]
    return flipped


def normalize_frame_landmarks(
    landmarks_21x3: np.ndarray,
    cfg: NormalizationConfig = NormalizationConfig(),
) -> np.ndarray:
    """Normalize a single frame of hand landmarks.

    Steps:
    1) Translate so wrist (idx 0) is at origin.
    2) Scale by max L2 distance from origin (scale-invariant).

    Input shape: (21,3)
    Output shape: (21,3)
    """
    if landmarks_21x3.shape != (NUM_LANDMARKS, DIMS):
        raise ValueError(f"Expected (21,3), got {landmarks_21x3.shape}")

    wrist = landmarks_21x3[0:1, :]
    centered = landmarks_21x3 - wrist

    dists = np.linalg.norm(centered[:, :2], axis=1)  # scale from x,y only
    scale = float(np.max(dists))
    if scale < cfg.eps:
        scale = 1.0

    normalized = centered / scale
    return normalized.astype(np.float32, copy=False)


def flatten_landmarks(landmarks_21x3: np.ndarray) -> np.ndarray:
    return landmarks_21x3.reshape(-1).astype(np.float32, copy=False)


def normalize_and_flatten_frame(landmarks_21x3: np.ndarray) -> np.ndarray:
    return flatten_landmarks(normalize_frame_landmarks(landmarks_21x3))


def sequence_to_model_input(
    landmarks_seq_30x21x3: np.ndarray,
    *,
    flip_x_raw01: bool = False,
) -> np.ndarray:
    """Convert a raw sequence into model input.

    Input: (T,21,3) raw MediaPipe normalized coords
    Output: (T,63) normalized + flattened
    """
    if landmarks_seq_30x21x3.ndim != 3:
        raise ValueError("Expected 3D array (T,21,3)")
    if landmarks_seq_30x21x3.shape[1:] != (NUM_LANDMARKS, DIMS):
        raise ValueError(f"Expected (*,21,3), got {landmarks_seq_30x21x3.shape}")

    seq = landmarks_seq_30x21x3
    if flip_x_raw01:
        seq = np.stack([flip_landmarks_x_raw01(fr) for fr in seq], axis=0)

    out = np.stack([normalize_and_flatten_frame(fr) for fr in seq], axis=0)
    return out.astype(np.float32, copy=False)


def resample_sequence_linear(
    landmarks_seq_tx21x3: np.ndarray,
    target_len: int,
) -> np.ndarray:
    """Time-warp augmentation via linear interpolation to target length."""
    t, n, d = landmarks_seq_tx21x3.shape
    if (n, d) != (NUM_LANDMARKS, DIMS):
        raise ValueError("Expected (T,21,3)")
    if t == target_len:
        return landmarks_seq_tx21x3.astype(np.float32, copy=False)

    x_old = np.linspace(0.0, 1.0, num=t, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)

    flat = landmarks_seq_tx21x3.reshape(t, -1)
    out = np.empty((target_len, flat.shape[1]), dtype=np.float32)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(x_new, x_old, flat[:, j]).astype(np.float32)
    return out.reshape(target_len, NUM_LANDMARKS, DIMS)


def add_gaussian_jitter(
    landmarks_seq_tx21x3: np.ndarray,
    sigma: float = 0.003,
) -> np.ndarray:
    noise = np.random.normal(0.0, sigma, size=landmarks_seq_tx21x3.shape).astype(
        np.float32
    )
    return (landmarks_seq_tx21x3.astype(np.float32, copy=False) + noise).astype(
        np.float32, copy=False
    )


def topk_softmax(probs: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(-probs)[:k]
    return idx, probs[idx]
