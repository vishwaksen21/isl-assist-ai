from __future__ import annotations

from typing import Any, Dict

from app.ml.model import LSTMClassifier
from app.ml.stgcn_model import STGCNClassifier
from app.ml.transformer_model import TransformerClassifier


def build_model_from_config(cfg: Dict[str, Any]):
    arch = str(cfg.get("arch", "lstm"))

    if arch == "lstm":
        return LSTMClassifier(
            input_size=int(cfg["input_size"]),
            hidden_size=int(cfg["hidden_size"]),
            num_layers=int(cfg["num_layers"]),
            num_classes=int(cfg["num_classes"]),
            dropout=float(cfg["dropout"]),
        )

    if arch == "transformer":
        return TransformerClassifier(
            input_size=int(cfg["input_size"]),
            d_model=int(cfg.get("d_model", 128)),
            nhead=int(cfg.get("nhead", 4)),
            num_layers=int(cfg.get("num_layers", 4)),
            dim_feedforward=int(cfg.get("dim_feedforward", 256)),
            num_classes=int(cfg["num_classes"]),
            dropout=float(cfg.get("dropout", 0.3)),
            max_len=int(cfg.get("max_len", 64)),
        )

    if arch == "stgcn":
        # hand-only
        return STGCNClassifier(
            num_classes=int(cfg["num_classes"]),
            dropout=float(cfg.get("dropout", 0.3)),
            hidden_channels=int(cfg.get("hidden_channels", 64)),
        )

    raise ValueError(f"Unknown arch: {arch}")
