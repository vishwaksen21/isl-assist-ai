# Custom ISL Dataset Structure (Keypoints)

This project trains on **MediaPipe hand landmarks** (21 points) as **keypoint coordinates**, not raw images.

## Recommended Structure

```
data/
  raw/                      # recorded sequences (keypoints)
    <LABEL_NAME>/
      sample_000001.npz
      sample_000002.npz
  splits/                   # optional cached splits
  README/
    STRUCTURE.md
```

Each `sample_*.npz` contains:
- `landmarks`: float32 array shaped `(30, 21, 3)` (30 frames, 21 landmarks × (x,y,z)) in MediaPipe normalized coordinates
- `label`: string label name

## Recording

Use `scripts/record_dataset.py` to record samples from your webcam.

## Notes

- Use consistent lighting/background.
- Record multiple users + angles.
- Prefer 50–200 samples per class for a baseline.

During training/inference, frames are converted to model input by:
1) translating wrist (landmark 0) to the origin
2) scaling by max hand radius (scale-invariant)
3) flattening to `(30, 63)` for the LSTM.
