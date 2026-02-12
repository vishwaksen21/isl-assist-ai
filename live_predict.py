from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import sys

import cv2
import mediapipe as mp
import numpy as np
import torch

# Allow running from repo root (so `import app.*` works)
sys.path.append(str(Path(__file__).resolve().parent / "backend"))

from app.ml.factory import build_model_from_config  # noqa: E402
from app.ml.preprocess import sequence_to_model_input  # noqa: E402


def load_labels(labels_path: str | Path) -> list[str]:
    """Load labels from artifacts.

    Supported formats:
    - {"labels": ["HELLO", ...]}  (current)
    - {"HELLO": 0, "YES": 1, ...} (legacy)
    - ["HELLO", "YES", ...]       (legacy)
    """
    p = Path(labels_path)
    payload = json.loads(p.read_text(encoding="utf-8"))

    if isinstance(payload, dict) and "labels" in payload and isinstance(payload["labels"], list):
        return [str(x) for x in payload["labels"]]

    if isinstance(payload, dict):
        # assume label->index mapping
        pairs = [(str(k), int(v)) for k, v in payload.items()]
        pairs.sort(key=lambda kv: kv[1])
        return [k for k, _ in pairs]

    if isinstance(payload, list):
        return [str(x) for x in payload]

    raise ValueError(f"Unsupported labels.json format: {type(payload)}")


# ---- Load checkpoint ----
# Explicit weights_only=False to avoid FutureWarning and because this checkpoint includes non-tensor config.
checkpoint = torch.load("artifacts/model.pt", map_location="cpu", weights_only=False)
cfg = checkpoint["config"]
state_dict = checkpoint["state_dict"]

# ---- Load labels ----
labels = load_labels("artifacts/labels.json")

# ---- Build model based on checkpoint config ----
model = build_model_from_config(cfg).to("cpu")
model.load_state_dict(state_dict, strict=True)
model.eval()

# ---- MediaPipe setup ----
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit(
        "ERROR: Could not open webcam (camera index 0).\n\n"
        "This is common in dev containers / remote environments where /dev/video0 is not exposed.\n"
        "Options:\n"
        "  1) Run locally on your host OS (not inside the container).\n"
        "  2) If using Docker directly, pass the device through (example):\n"
        "       docker run --device=/dev/video0 ...\n"
        "  3) Use the browser-based webcam pipeline instead:\n"
        "       - start backend:  cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000\n"
        "       - start frontend: cd frontend && python -m http.server 5173\n"
        "       - open http://localhost:5173\n"
    )
sequence = deque(maxlen=30)

print("Press Q to quit | Press R to reset buffer")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = []

        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])

        sequence.append(landmarks)

        if len(sequence) == 30:
            # Convert raw MediaPipe landmarks (30,21,3) -> model input (30,63)
            raw_seq = np.asarray(sequence, dtype=np.float32)
            x = sequence_to_model_input(raw_seq, flip_x_raw01=False)  # (30,63)
            input_data = torch.from_numpy(x).unsqueeze(0)  # (1,30,63)

            with torch.no_grad():
                logits = model(input_data)
                pred = int(torch.argmax(logits, dim=1).item())
                label = labels[pred] if 0 <= pred < len(labels) else "?"

            cv2.putText(frame, label, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

    cv2.imshow("Live Prediction", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        sequence.clear()
        print("Buffer cleared")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
