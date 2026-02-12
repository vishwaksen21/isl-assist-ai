# Training on macOS (Apple Silicon M1/M2)

PyTorch can use MPS acceleration.

1) Create venv and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Train:

```bash
python scripts/train.py --data-dir data/raw --epochs 30
```

The training script automatically selects device:
- CUDA (if available)
- else MPS (on Apple Silicon)
- else CPU
