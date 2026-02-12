from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np


HAND_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a unique synthetic ISL keypoint dataset (for pipeline testing)"
    )
    p.add_argument("--out-dir", type=str, default="data/raw")
    p.add_argument(
        "--labels",
        type=str,
        default="HELLO,YES,THANK_YOU",
        help="Comma-separated labels",
    )
    p.add_argument("--samples-per-label", type=int, default=80)
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _base_hand_pose(rng: np.random.RandomState) -> np.ndarray:
    """Create a plausible base hand pose in MediaPipe normalized coords."""
    # wrist near center
    wrist = np.array([0.5, 0.65, 0.0], dtype=np.float32)

    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = wrist

    # Five fingers with 4 joints each
    # Rough directions for thumb/index/middle/ring/pinky
    dirs = {
        "thumb": np.array([-0.10, -0.06, 0.00], dtype=np.float32),
        "index": np.array([-0.06, -0.16, 0.00], dtype=np.float32),
        "middle": np.array([0.00, -0.18, 0.00], dtype=np.float32),
        "ring": np.array([0.06, -0.16, 0.00], dtype=np.float32),
        "pinky": np.array([0.10, -0.12, 0.00], dtype=np.float32),
    }
    chains = {
        "thumb": [1, 2, 3, 4],
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20],
    }

    for name, idxs in chains.items():
        d = dirs[name]
        # randomize per-finger spread slightly
        d = d + rng.normal(0, 0.01, size=(3,)).astype(np.float32)
        d[2] = 0.0
        prev = wrist
        for k, i in enumerate(idxs):
            scale = (k + 1) * 1.0
            pts[i] = prev + d * 0.45 * scale
            prev = pts[i]

    pts[:, 0] = np.clip(pts[:, 0], 0.05, 0.95)
    pts[:, 1] = np.clip(pts[:, 1], 0.05, 0.95)
    return pts


def _apply_class_motion(
    seq: np.ndarray, label: str, rng: np.random.RandomState
) -> np.ndarray:
    """Inject class-specific temporal patterns so classes are separable."""
    t = seq.shape[0]
    out = seq.copy()

    # motion components
    phase = rng.uniform(0, 2 * math.pi)
    amp = rng.uniform(0.008, 0.02)
    wrist = out[:, 0:1, :]

    if label.upper() == "HELLO":
        # waving: oscillate x for fingertips
        tip_idxs = [4, 8, 12, 16, 20]
        for ti in range(t):
            dx = amp * math.sin(2 * math.pi * ti / t + phase)
            out[ti, tip_idxs, 0] += dx

    elif label.upper() == "YES":
        # curling gesture: bring fingertips towards wrist (relative shape change)
        tip_idxs = [4, 8, 12, 16, 20]
        mid_idxs = [3, 7, 11, 15, 19]
        for ti in range(t):
            f = 0.75 + 0.20 * math.sin(2 * math.pi * ti / t + phase)  # 0.55..0.95
            # pull tips and near-tips towards wrist in x/y (not the wrist itself)
            out[ti, tip_idxs, :2] = wrist[ti, 0, :2] + (out[ti, tip_idxs, :2] - wrist[ti, 0, :2]) * f
            out[ti, mid_idxs, :2] = wrist[ti, 0, :2] + (out[ti, mid_idxs, :2] - wrist[ti, 0, :2]) * (0.85 + (f - 0.75) * 0.5)

    elif label.upper() == "THANK_YOU":
        # pinch gesture: thumb tip approaches index tip, plus slight forward z
        thumb_tip = 4
        index_tip = 8
        for ti in range(t):
            f = 0.35 + 0.25 * math.sin(2 * math.pi * ti / t + phase)  # closeness factor
            v = out[ti, index_tip, :2] - out[ti, thumb_tip, :2]
            out[ti, thumb_tip, :2] += v * f
            out[ti, index_tip, :2] -= v * f

            dz = amp * math.sin(math.pi * ti / t)
            out[ti, [thumb_tip, index_tip], 2] += dz

    else:
        # generic small circular motion
        # move fingertips in a circle around their own MCP (relative)
        tip_idxs = [4, 8, 12, 16, 20]
        base_idxs = [2, 6, 10, 14, 18]
        for ti in range(t):
            dx = amp * math.sin(2 * math.pi * ti / t + phase)
            dy = amp * math.cos(2 * math.pi * ti / t + phase)
            out[ti, tip_idxs, 0] = out[ti, base_idxs, 0] + (out[ti, tip_idxs, 0] - out[ti, base_idxs, 0]) + dx
            out[ti, tip_idxs, 1] = out[ti, base_idxs, 1] + (out[ti, tip_idxs, 1] - out[ti, base_idxs, 1]) + dy

    return out


def _sample_sequence(label: str, seq_len: int, rng: np.random.RandomState) -> np.ndarray:
    base = _base_hand_pose(rng)

    # per-sample affine jitter
    tx = rng.normal(0.0, 0.02)
    ty = rng.normal(0.0, 0.02)
    base[:, 0] = np.clip(base[:, 0] + tx, 0.0, 1.0)
    base[:, 1] = np.clip(base[:, 1] + ty, 0.0, 1.0)

    seq = np.stack([base for _ in range(seq_len)], axis=0).astype(np.float32)
    seq = _apply_class_motion(seq, label, rng)

    # landmark noise
    seq += rng.normal(0.0, 0.003, size=seq.shape).astype(np.float32)

    # clamp x,y to [0,1], keep z small
    seq[:, :, 0] = np.clip(seq[:, :, 0], 0.0, 1.0)
    seq[:, :, 1] = np.clip(seq[:, :, 1], 0.0, 1.0)
    seq[:, :, 2] = np.clip(seq[:, :, 2], -0.25, 0.25)

    return seq.astype(np.float32)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    if not labels:
        raise SystemExit("No labels provided")

    rng = np.random.RandomState(args.seed)

    total = 0
    for label in labels:
        label_dir = out_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        for i in range(args.samples_per_label):
            seq = _sample_sequence(label, args.seq_len, rng)
            ts = int(time.time() * 1000)
            path = label_dir / f"demo_{label}_{ts}_{i:06d}.npz"
            np.savez_compressed(path, landmarks=seq, label=label)
            total += 1

    print(f"Generated {total} samples under {out_dir}")
    print("Next steps:")
    print(f"  python validate_dataset.py --data-dir {out_dir}")
    print(f"  python train.py --data-dir {out_dir} --arch transformer")


if __name__ == "__main__":
    main()
