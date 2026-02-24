# ISL Assist AI – Real-Time Indian Sign Language Recognition System

Web app that captures ISL gestures from a webcam and converts them to **text + speech** in real time.

## Tech Stack

- Frontend: HTML + CSS + JavaScript, WebRTC (`getUserMedia`) + Web Speech API
- Backend: Python + FastAPI
- CV/AI: OpenCV + MediaPipe (21 hand landmarks) + PyTorch (LSTM)

## Project Structure

```
.
├─ backend/                  # FastAPI app (MediaPipe + model inference)
├─ frontend/                 # Static HTML/CSS/JS UI
├─ scripts/                  # Data recording + training
├─ data/
│  └─ raw/                   # Your recorded keypoint sequences (generated)
├─ artifacts/                # Trained model + labels + metrics (generated)
├─ requirements.txt
└─ Dockerfile
```

## 1) Setup (Local)

### Prerequisites

- Python 3.10+ (3.11 recommended)
- A webcam

### Install

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Quick Run (one command)

If you already installed dependencies and have trained artifacts, you can start
both backend + frontend together:

```bash
bash scripts/run_all.sh
```

Then open http://localhost:5173

Note: the backend requires `artifacts/model.pt` + `artifacts/labels.json`. If
they are missing, train first (see "Model Training").

## Keep backend always-on (macOS optional)

If you want the backend to start automatically on login (so you don't manually
start it every time), you can install a LaunchAgent:

```bash
chmod +x scripts/macos/install_backend_launchagent.sh
./scripts/macos/install_backend_launchagent.sh
```

This expects your Python environment at `.venv/`.
Logs: `tail -f /tmp/isl-assist-ai-backend.log`

Troubleshooting (Linux/dev containers):
- If you see `ImportError: libGL.so.1: cannot open shared object file`, you’re using GUI OpenCV wheels.
	- Recommended: use the repo default `opencv-python-headless` by reinstalling deps:
		- `pip uninstall -y opencv-python`
		- `pip install -r requirements.txt`
	- Or install OS libs: `sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0`

## 2) Dataset (Custom ISL)

This system trains on **keypoint coordinates**, not images.

### What gets saved?

Each sample is a `.npz` file containing:

- `landmarks`: `(30, 21, 3)` MediaPipe normalized hand landmarks per frame
- `label`: string sign label

See [data/README/STRUCTURE.md](data/README/STRUCTURE.md).

### Record data using webcam

#### Option A (recommended in dev containers): record via the browser UI

In remote/dev-container environments, Python/OpenCV often cannot access your webcam (and `cv2.imshow` may fail). The web UI does not have this limitation.

- Start the backend + frontend (see sections below).
- Click **Start** to begin streaming frames (this also creates a session).
- Perform a sign and keep your hand in view until the status reaches `Buffering frames (30/30)` (or it switches to `Running`).
- In **Dataset Recorder (Phase-1)**, enter the label (e.g., `HELLO`) and click **Save Sample**.

This writes an `.npz` to `data/raw/<LABEL>/sample_<timestamp>.npz`.

#### Option B: record using the OpenCV script (local machine)

If you run locally on a machine with direct camera + GUI support, you can use the OpenCV recorder.
Note: this requires GUI-capable OpenCV (`opencv-python`) and OS GUI libraries.

Record 50 samples of label `HELLO`:

```bash
python record_dataset.py --label HELLO --samples 50
# or: python scripts/record_dataset.py --label HELLO --samples 50
```

Phase-1 quickstart (recommended):

```bash
python record_dataset.py --label HELLO --samples 80
python record_dataset.py --label THANK_YOU --samples 80
python record_dataset.py --label YES --samples 80
python validate_dataset.py --data-dir data/raw
```

### If you don’t have real samples yet (demo dataset)

To quickly test the full pipeline end-to-end, generate a **unique synthetic dataset** (hand landmarks only):

```bash
python generate_demo_dataset.py --out-dir data/raw --labels HELLO,YES,THANK_YOU --samples-per-label 80
python validate_dataset.py --data-dir data/raw
python train.py --data-dir data/raw --arch transformer
```

If you already generated a demo dataset earlier, delete `data/raw/*/*.npz` and regenerate to get the latest patterns.

For a quick sanity check that training is working end-to-end, you can also try:

```bash
python train.py --data-dir data/raw --arch lstm
```

This demo dataset is only for verifying the pipeline; for real accuracy, record real samples.

## Phase-1: Demo → Real Webcam Dataset

Once the demo dataset trains well, treat the pipeline as validated and switch to real data.

Checklist:
- Pick the best architecture once (LSTM vs Transformer vs ST-GCN).
- Freeze hyperparameters using a config JSON under `configs/`.
- Collect a balanced real dataset (same number of samples per label).
- Re-train using the same frozen config; only tune if metrics show a clear issue.

Example (Transformer, frozen):

```bash
# 1) Start fresh real dataset (optional)
rm -f data/raw/*/*.npz

# 2) Record real samples (repeat per label)
# If `record_dataset.py` fails in your environment, use the browser UI Dataset Recorder instead.
python record_dataset.py --label HELLO --samples 120
python record_dataset.py --label THANK_YOU --samples 120
python record_dataset.py --label YES --samples 120

# 3) Validate
python validate_dataset.py --data-dir data/raw

# 4) Train using frozen config
python train.py --data-dir data/raw --config configs/phase1_transformer.json
```

Controls in the recorder window:
- `SPACE`: start/stop recording current sample (collects until 30 valid frames)
- `R`: reset current sample
- `Q` / `ESC`: quit

Output folder example:

```
data/raw/HELLO/sample_....npz
```

### Validate/clean the dataset

Before training (especially if you recorded over multiple sessions), validate your dataset:

```bash
python scripts/validate_dataset.py --data-dir data/raw
# or (convenience wrapper)
python validate_dataset.py --data-dir data/raw
```

This checks:
- file integrity and required keys
- shape is exactly `(30, 21, 3)`
- no NaN/Inf values
- label matches folder name

To delete invalid samples automatically:

```bash
python scripts/validate_dataset.py --data-dir data/raw --delete-invalid
```

### Data augmentation suggestions (implemented in training)

- **Speed variation**: time-warp sequences by resampling to different lengths
- **Flipping**: horizontal flip in raw MediaPipe x-space (`x := 1 - x`)
- **Jitter**: small Gaussian noise on keypoints

## 3) Model Training (LSTM)

### Core pipeline

- Extract 21 landmarks per frame (MediaPipe)
- Use keypoints (not raw images)
- Normalize per frame before training:
	1) translate wrist (landmark 0) to origin
	2) scale by max hand radius
	3) flatten to `(30, 63)` features
- LSTM sequence model:
	- Input: 30 frames × 63
	- Output: sign class
	- Loss: `CrossEntropyLoss`
	- Optimizer: `Adam`
	- Dropout enabled
	- Early stopping on validation loss

### Train

```bash
# LSTM (default)
python scripts/train.py --data-dir data/raw --epochs 30 --batch-size 64 --arch lstm
# or (convenience wrapper)
python train.py --data-dir data/raw --epochs 30 --batch-size 64 --arch lstm

# Transformer encoder
python scripts/train.py --data-dir data/raw --epochs 30 --batch-size 64 --arch transformer

# ST-GCN (graph model on hand topology)
python scripts/train.py --data-dir data/raw --epochs 30 --batch-size 64 --arch stgcn
```

Device selection during training:
- CUDA if available
- else Apple Silicon MPS if available
- else CPU

Generated artifacts:

- `artifacts/model.pt` (best checkpoint)
- `artifacts/labels.json`
- `artifacts/train_history.json`
- `artifacts/metrics.json` (classification report + confusion matrix)
- `artifacts/confusion_matrix.png`

## 4) Run Backend (Real-time Prediction API)

The backend:
- accepts webcam frames
- runs MediaPipe on the backend
- buffers 30 frames per session
- runs the LSTM and returns prediction + confidence

Start API:

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `GET /api/session`
- `POST /api/predict/frame` (multipart: `frame`, `session_id`)
- `POST /api/session/reset?session_id=...`
- `POST /api/feedback` (JSON: `session_id`, `correct_label`) saves last 30-frame sequence under the corrected label

## 5) Run Frontend (WebRTC + TTS)

Serve the static site:

```bash
cd frontend
python -m http.server 5173
```

Open:
- http://localhost:5173

In the UI:
- Set Backend URL to `http://localhost:8000`
- Click **Start**
- The app shows **prediction**, **confidence**, and builds a text output
- Click **Speak** to use Web Speech API text-to-speech

### Duplicate filtering + sentence formation

- A word is appended only if it differs from the previous word.
- To improve isolated-sign accuracy, the UI uses a short stability window: it appends a word only when the same label is consistently predicted across recent frames and confidence >= 0.60.

## Accuracy tips (Phase-1)

- Record balanced data per class (similar sample counts).
- Training uses class-weighted loss by default to help with imbalance. Disable via `--no-class-weights`.
- Use the saved plots in `artifacts/` (`confusion_matrix.png`, `training_curves.png`) to diagnose overfitting.

Note: Achieving >90% accuracy depends on label set difficulty and data quality (lighting, signer diversity, and balanced samples). The pipeline is designed to make that target realistic for a focused isolated-sign vocabulary.

## 6) Deployment Guide

### Option A: Docker (simple)

1) Train locally and put artifacts into `artifacts/` (or mount them in production)
2) Build and run:

```bash
docker build -t isl-assist-ai .
docker run --rm -p 8000:8000 isl-assist-ai
```

Notes:
- The image includes an empty `artifacts/` directory by default; if you haven't trained yet, the API will return **503** until `artifacts/model.pt` and `artifacts/labels.json` exist.
- For production, prefer mounting your trained `artifacts/` into the container.

### Option B: VM/Server (no Docker)

- Run behind a reverse proxy (e.g., Nginx) and start with:

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Notes:
- OpenCV may require system packages (`libgl1`, `libglib2.0-0`) depending on your OS.
- The model runs on CPU by default; if CUDA is available, training uses it automatically.

## Troubleshooting

- **503 Model artifacts not found**: run training first so `artifacts/model.pt` and `artifacts/labels.json` exist.
- **No hand detected**: improve lighting, keep hand inside frame, reduce motion blur.
