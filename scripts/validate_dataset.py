from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate recorded ISL keypoint dataset (*.npz)")
    p.add_argument("--data-dir", type=str, default="data/raw")
    p.add_argument(
        "--delete-invalid",
        action="store_true",
        help="Delete invalid/corrupted samples (default: only report)",
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=30,
        help="Expected sequence length (frames)",
    )
    return p.parse_args()


def iter_samples(data_dir: Path) -> List[Path]:
    return sorted([p for p in data_dir.rglob("*.npz") if p.is_file()])


def validate_file(path: Path, seq_len: int) -> Tuple[bool, str, str]:
    """Returns (ok, label, reason)."""
    try:
        item = np.load(path, allow_pickle=False)
    except Exception as e:
        return False, "", f"load_error: {e}"

    if "landmarks" not in item or "label" not in item:
        return False, "", "missing_keys"

    label = str(item["label"])
    landmarks = item["landmarks"]

    if landmarks.ndim != 3 or landmarks.shape[1:] != (21, 3):
        return False, label, f"bad_shape: {getattr(landmarks, 'shape', None)}"

    if int(landmarks.shape[0]) != int(seq_len):
        return False, label, f"bad_seq_len: {landmarks.shape[0]}"

    if not np.isfinite(landmarks).all():
        return False, label, "non_finite_values"

    # Optional: check that label matches parent folder if structured that way
    parent_label = path.parent.name
    if parent_label and parent_label != label:
        return False, label, f"label_folder_mismatch: folder={parent_label} label={label}"

    return True, label, "ok"


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    # If missing, create it so the next step can be demo-generation or recording.
    data_dir.mkdir(parents=True, exist_ok=True)

    files = iter_samples(data_dir)
    if not files:
        raise SystemExit(
            f"No .npz samples found under {data_dir}\n"
            "Record samples first, e.g.:\n"
            "  python record_dataset.py --label HELLO --samples 50\n"
            "Or generate a demo dataset to test the pipeline:\n"
            "  python generate_demo_dataset.py --out-dir data/raw --labels HELLO,YES,THANK_YOU --samples-per-label 80\n"
            "Then validate again: python validate_dataset.py --data-dir data/raw"
        )

    ok_files: List[Path] = []
    bad_files: List[Tuple[Path, str]] = []
    label_counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()

    for f in files:
        ok, label, reason = validate_file(f, args.seq_len)
        reasons[reason] += 1
        if ok:
            ok_files.append(f)
            label_counts[label] += 1
        else:
            bad_files.append((f, reason))

    print(f"Total samples: {len(files)}")
    print(f"Valid samples: {len(ok_files)}")
    print(f"Invalid samples: {len(bad_files)}")

    print("\nInvalid reasons:")
    for r, c in reasons.most_common():
        if r != "ok":
            print(f"  {r}: {c}")

    if bad_files:
        print("\nFirst 20 invalid files:")
        for f, reason in bad_files[:20]:
            print(f"  {f} -> {reason}")

    print("\nClass distribution (valid only):")
    for label, c in label_counts.most_common():
        print(f"  {label}: {c}")

    if label_counts:
        min_c = min(label_counts.values())
        max_c = max(label_counts.values())
        print(f"\nImbalance ratio (max/min): {max_c}/{min_c} = {max_c / max(1, min_c):.2f}")

    if args.delete_invalid and bad_files:
        for f, _ in bad_files:
            try:
                f.unlink()
            except Exception:
                pass
        print(f"\nDeleted {len(bad_files)} invalid files.")


if __name__ == "__main__":
    main()
