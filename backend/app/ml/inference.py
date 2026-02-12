from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from app.ml.factory import build_model_from_config


class SignPredictor:
    def __init__(
        self,
        model_path: str | Path,
        labels_path: str | Path,
        *,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        model_path = Path(model_path)
        labels_path = Path(labels_path)

        with labels_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self.labels: List[str] = payload["labels"]

        ckpt = torch.load(model_path, map_location=self.device)
        self.model = build_model_from_config(ckpt["config"]).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, seq_30x63: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        x = torch.from_numpy(seq_30x63.astype(np.float32, copy=False)).unsqueeze(0)
        x = x.to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        idx = int(np.argmax(probs))
        label = self.labels[idx]
        conf = float(probs[idx])
        dist = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
        return label, conf, dist
