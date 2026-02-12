from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Allow running from repo root
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "backend"))

from app.ml.model import LSTMClassifier  # noqa: E402

# Backwards-compatibility exports
# Some local scripts (e.g. live_predict.py) import these symbols from scripts.train.
LSTMModel = LSTMClassifier

try:  # pragma: no cover
    from app.ml.transformer_model import TransformerClassifier as TransformerModel  # noqa: E402
except Exception:  # pragma: no cover
    TransformerModel = None  # type: ignore[assignment]

try:  # pragma: no cover
    from app.ml.stgcn_model import STGCNClassifier as STGCNModel  # noqa: E402
except Exception:  # pragma: no cover
    STGCNModel = None  # type: ignore[assignment]

__all__ = [
    "LSTMModel",
    "TransformerModel",
    "STGCNModel",
]
from app.ml.preprocess import (  # noqa: E402
    add_gaussian_jitter,
    resample_sequence_linear,
    sequence_to_model_input,
)


@dataclass
class TrainConfig:
    seq_len: int = 30
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    early_stop_patience: int = 7

    # Architecture
    arch: str = "lstm"  # lstm|transformer|stgcn
    # Transformer
    d_model: int = 128
    nhead: int = 4
    dim_feedforward: int = 256
    max_len: int = 64
    # ST-GCN
    hidden_channels: int = 64

    # Augmentation
    aug_time_warp: bool = True
    aug_flip: bool = True
    aug_jitter: bool = True

    # Split
    seed: int = 42
    val_ratio: float = 0.15
    test_ratio: float = 0.15


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _apply_overrides(cfg: TrainConfig, overrides: Dict) -> TrainConfig:
    data = cfg.__dict__.copy()
    for k, v in overrides.items():
        if k == "use_class_weights":
            # handled separately
            continue
        if k in data:
            data[k] = v
    return TrainConfig(**data)


class ISLKeypointDataset(Dataset):
    def __init__(
        self,
        files: List[Path],
        label_to_id: Dict[str, int],
        cfg: TrainConfig,
        training: bool,
    ) -> None:
        self.files = files
        self.label_to_id = label_to_id
        self.cfg = cfg
        self.training = training

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = np.load(self.files[idx], allow_pickle=False)
        landmarks = item["landmarks"].astype(np.float32)  # (T,21,3)
        label = str(item["label"])

        # Basic cleaning: ensure length
        if landmarks.shape[0] != self.cfg.seq_len:
            landmarks = resample_sequence_linear(landmarks, self.cfg.seq_len)

        # Augmentations (operate in raw [0,1] MediaPipe coord space)
        flip = False
        if self.training:
            if self.cfg.aug_time_warp and np.random.rand() < 0.5:
                # Warp to random length then resample back
                target = int(np.random.choice([20, 24, 28, 30, 34, 38, 42]))
                warped = resample_sequence_linear(landmarks, target)
                landmarks = resample_sequence_linear(warped, self.cfg.seq_len)

            if self.cfg.aug_jitter and np.random.rand() < 0.5:
                landmarks = add_gaussian_jitter(landmarks, sigma=0.002)

            if self.cfg.aug_flip and np.random.rand() < 0.5:
                flip = True

        x = sequence_to_model_input(landmarks, flip_x_raw01=flip)

        y = self.label_to_id[label]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def list_samples(data_dir: Path) -> List[Path]:
    return sorted([p for p in data_dir.rglob("*.npz") if p.is_file()])


def build_label_map(files: List[Path]) -> List[str]:
    labels = set()
    for f in files:
        item = np.load(f, allow_pickle=False)
        labels.add(str(item["label"]))
    return sorted(labels)


def split_files(files: List[Path], cfg: TrainConfig) -> Tuple[List[Path], List[Path], List[Path]]:
    # Stratified split by label for stable evaluation
    y = []
    for f in files:
        item = np.load(f, allow_pickle=False)
        y.append(str(item["label"]))

    train_files, tmp_files, y_train, y_tmp = train_test_split(
        files,
        y,
        test_size=(cfg.val_ratio + cfg.test_ratio),
        random_state=cfg.seed,
        stratify=y,
    )

    # Split tmp into val and test
    if cfg.val_ratio + cfg.test_ratio == 0:
        return train_files, [], []

    rel_test = cfg.test_ratio / (cfg.val_ratio + cfg.test_ratio)
    val_files, test_files, _, _ = train_test_split(
        tmp_files,
        y_tmp,
        test_size=rel_test,
        random_state=cfg.seed,
        stratify=y_tmp,
    )

    return list(train_files), list(val_files), list(test_files)


def eval_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    crit: nn.Module,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            total_loss += float(loss.item()) * x.size(0)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    return avg_loss, acc, np.array(y_true), np.array(y_pred)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM on ISL keypoints")
    p.add_argument("--data-dir", default="data/raw", type=str)
    p.add_argument("--artifacts", default="artifacts", type=str)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--arch",
        type=str,
        default="lstm",
        choices=["lstm", "transformer", "stgcn"],
        help="Model architecture",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON config to freeze hyperparameters (see configs/phase1_*.json)",
    )
    p.add_argument(
        "--save-config",
        type=str,
        default=None,
        help="Write the effective config used for this run to a JSON file",
    )
    p.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class-weighted loss (enabled by default)",
    )
    return p.parse_args()


def compute_class_weights(files: List[Path], label_to_id: Dict[str, int]) -> torch.Tensor:
    counts = np.zeros((len(label_to_id),), dtype=np.int64)
    for f in files:
        item = np.load(f, allow_pickle=False)
        label = str(item["label"])
        counts[label_to_id[label]] += 1
    counts = np.maximum(counts, 1)
    # inverse frequency, normalized
    weights = counts.sum() / (len(counts) * counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, arch=args.arch)

    use_class_weights = not args.no_class_weights
    if args.config:
        overrides = _load_json(Path(args.config))
        cfg = _apply_overrides(cfg, overrides)
        if "arch" in overrides:
            cfg = _apply_overrides(cfg, {"arch": overrides["arch"]})
        if "use_class_weights" in overrides:
            use_class_weights = bool(overrides["use_class_weights"])

    data_dir = Path(args.data_dir)
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)

    if args.save_config:
        out_cfg = cfg.__dict__.copy()
        out_cfg["use_class_weights"] = use_class_weights
        Path(args.save_config).write_text(json.dumps(out_cfg, indent=2), encoding="utf-8")

    files = list_samples(data_dir)
    if not files:
        raise SystemExit(
            f"No samples found under {data_dir} (expected *.npz)\n"
            "Record samples first, e.g.:\n"
            "  python record_dataset.py --label HELLO --samples 50\n"
            "Or generate a demo dataset to test the pipeline:\n"
            "  python generate_demo_dataset.py --out-dir data/raw --labels HELLO,YES,THANK_YOU --samples-per-label 80\n"
            "Then train again."
        )

    labels = build_label_map(files)
    label_to_id = {l: i for i, l in enumerate(labels)}
    num_classes = len(labels)

    train_files, val_files, test_files = split_files(files, cfg)

    train_ds = ISLKeypointDataset(train_files, label_to_id, cfg, training=True)
    val_ds = ISLKeypointDataset(val_files, label_to_id, cfg, training=False)
    test_ds = ISLKeypointDataset(test_files, label_to_id, cfg, training=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if cfg.arch == "lstm":
        model = LSTMClassifier(
            input_size=63,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_classes=num_classes,
            dropout=cfg.dropout,
        ).to(device)
        model_cfg = {
            "arch": "lstm",
            "input_size": 63,
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "num_classes": num_classes,
            "dropout": cfg.dropout,
        }
    elif cfg.arch == "transformer":
        from app.ml.transformer_model import TransformerClassifier

        model = TransformerClassifier(
            input_size=63,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            num_classes=num_classes,
            dropout=cfg.dropout,
            max_len=cfg.max_len,
        ).to(device)
        model_cfg = {
            "arch": "transformer",
            "input_size": 63,
            "d_model": cfg.d_model,
            "nhead": cfg.nhead,
            "num_layers": cfg.num_layers,
            "dim_feedforward": cfg.dim_feedforward,
            "num_classes": num_classes,
            "dropout": cfg.dropout,
            "max_len": cfg.max_len,
        }
    else:
        from app.ml.stgcn_model import STGCNClassifier

        model = STGCNClassifier(
            num_classes=num_classes,
            dropout=cfg.dropout,
            hidden_channels=cfg.hidden_channels,
        ).to(device)
        model_cfg = {
            "arch": "stgcn",
            "input_size": 63,
            "hidden_channels": cfg.hidden_channels,
            "num_classes": num_classes,
            "dropout": cfg.dropout,
        }

    if not use_class_weights:
        crit = nn.CrossEntropyLoss()
    else:
        w = compute_class_weights(train_files, label_to_id).to(device)
        crit = nn.CrossEntropyLoss(weight=w)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_loss = float("inf")
    patience = 0

    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        y_true_epoch, y_pred_epoch = [], []

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}"):
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            optim.step()

            running_loss += float(loss.item()) * x.size(0)
            preds = torch.argmax(logits, dim=1)
            y_true_epoch.extend(y.detach().cpu().numpy())
            y_pred_epoch.extend(preds.detach().cpu().numpy())

        train_loss = running_loss / max(1, len(train_loader.dataset))
        train_acc = accuracy_score(y_true_epoch, y_pred_epoch) if y_true_epoch else 0.0

        val_loss, val_acc, _, _ = eval_model(model, val_loader, device, crit)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience = 0

            ckpt = {
                "state_dict": model.state_dict(),
                "config": model_cfg,
            }
            torch.save(ckpt, artifacts / "model.pt")
            (artifacts / "labels.json").write_text(
                json.dumps({"labels": labels}, indent=2),
                encoding="utf-8",
            )
            (artifacts / "train_history.json").write_text(
                json.dumps(history, indent=2),
                encoding="utf-8",
            )
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print("Early stopping triggered")
                break

    # Load best checkpoint for final test eval
    best = torch.load(artifacts / "model.pt", map_location=device)
    model.load_state_dict(best["state_dict"], strict=True)

    test_loss, test_acc, y_true, y_pred = eval_model(model, test_loader, device, nn.CrossEntropyLoss())
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    macro_f1 = float(report.get("macro avg", {}).get("f1-score", 0.0))
    macro_precision = float(report.get("macro avg", {}).get("precision", 0.0))
    macro_recall = float(report.get("macro avg", {}).get("recall", 0.0))

    (artifacts / "metrics.json").write_text(
        json.dumps(
            {
                "test_loss": test_loss,
                "test_acc": test_acc,
                "macro_f1": macro_f1,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "classification_report": report,
                "confusion_matrix": cm.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))
    print(
        f"Macro: precision={macro_precision:.3f} recall={macro_recall:.3f} f1={macro_f1:.3f}"
    )
    print("Confusion matrix:")
    print(cm)

    # Save confusion matrix image
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(artifacts / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    # Save training curves
    epochs = [h["epoch"] for h in history]
    tr_loss = [h["train_loss"] for h in history]
    va_loss = [h["val_loss"] for h in history]
    tr_acc = [h["train_acc"] for h in history]
    va_acc = [h["val_acc"] for h in history]

    fig2 = plt.figure(figsize=(9, 4))
    ax1 = fig2.add_subplot(121)
    ax1.plot(epochs, tr_loss, label="train")
    ax1.plot(epochs, va_loss, label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("epoch")
    ax1.legend()

    ax2 = fig2.add_subplot(122)
    ax2.plot(epochs, tr_acc, label="train")
    ax2.plot(epochs, va_acc, label="val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("epoch")
    ax2.legend()

    fig2.tight_layout()
    fig2.savefig(artifacts / "training_curves.png", dpi=200)
    plt.close(fig2)


if __name__ == "__main__":
    main()
