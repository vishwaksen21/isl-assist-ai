import cv2
import torch
import numpy as np
import json
from collections import deque
import mediapipe as mp

# ---- Import model builders from train.py ----
from scripts.train import LSTMModel, TransformerModel, STGCNModel

# ---- Load checkpoint ----
checkpoint = torch.load("artifacts/model.pt", map_location="cpu")
cfg = checkpoint["config"]
state_dict = checkpoint["state_dict"]

# ---- Load labels ----
with open("artifacts/labels.json", "r") as f:
    label_to_index = json.load(f)

index_to_label = {v: k for k, v in label_to_index.items()}
num_classes = len(label_to_index)

# ---- Build model based on architecture ----
arch = cfg["arch"]

if arch == "lstm":
    model = LSTMModel(num_classes=num_classes)
elif arch == "transformer":
    model = TransformerModel(num_classes=num_classes)
elif arch == "stgcn":
    model = STGCNModel(num_classes=num_classes)
else:
    raise ValueError("Unknown architecture")

model.load_state_dict(state_dict)
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
            input_data = np.array(sequence)
            input_data = torch.tensor(input_data, dtype=torch.float32)
            input_data = input_data.unsqueeze(0)

            with torch.no_grad():
                output = model(input_data)
                pred = torch.argmax(output, dim=1).item()
                label = index_to_label[pred]

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
