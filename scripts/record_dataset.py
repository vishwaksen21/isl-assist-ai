from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# Allow running from repo root
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.mediapipe_service import MediaPipeHandLandmarker  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record ISL keypoint sequences")
    p.add_argument("--label", required=True, help="Sign label name")
    p.add_argument("--samples", type=int, default=50, help="Number of samples")
    p.add_argument("--seq-len", type=int, default=30, help="Frames per sample")
    p.add_argument(
        "--out-dir",
        type=str,
        default="data/raw",
        help="Output directory (will create label subfolder)",
    )
    p.add_argument("--camera", type=int, default=0, help="Webcam index")
    p.add_argument("--min-frames", type=int, default=30, help="Alias for seq-len")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seq_len = int(args.seq_len)
    out_dir = Path(args.out_dir) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    landmarker = MediaPipeHandLandmarker()

    sample_idx = 0
    recording = False
    collected = []

    print("Controls:")
    print("  SPACE  start/stop recording current sample")
    print("  R      reset current sample")
    print("  Q/ESC  quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # mirror preview (recording still uses mirrored)
        landmarks = landmarker.extract_first_hand_landmarks(frame)

        overlay = frame.copy()
        status = f"Label={args.label} Sample={sample_idx+1}/{args.samples} "
        status += f"Rec={'YES' if recording else 'NO'} Frames={len(collected)}/{seq_len}"

        if landmarks is None:
            cv2.putText(
                overlay,
                "No hand detected",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
        else:
            cv2.putText(
                overlay,
                "Hand detected",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            if recording:
                collected.append(landmarks)

        cv2.putText(
            overlay,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("ISL Assist AI - Recorder", overlay)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("r"), ord("R")):
            collected = []
            recording = False
        if key == 32:  # space
            recording = not recording

        if recording and len(collected) >= seq_len:
            arr = np.stack(collected[:seq_len], axis=0).astype(np.float32)
            ts = int(time.time() * 1000)
            path = out_dir / f"sample_{ts}_{sample_idx:06d}.npz"
            np.savez_compressed(path, landmarks=arr, label=args.label)
            print(f"Saved: {path}")

            sample_idx += 1
            collected = []
            recording = False

            if sample_idx >= args.samples:
                print("Done.")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
