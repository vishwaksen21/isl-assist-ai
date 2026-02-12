from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.routes.predict import seq_buffer


router = APIRouter(tags=["dataset"])


class SaveSampleIn(BaseModel):
    session_id: str
    label: str = Field(..., min_length=1)


@router.post("/dataset/save")
def save_dataset_sample(payload: SaveSampleIn) -> Dict[str, Any]:
    """Save the latest 30-frame buffered sequence under `data/raw/<label>/...npz`.

    Intended for Phase-1 data collection using the browser webcam:
    - frontend streams frames to /api/predict/frame (fills buffer)
    - user clicks Save -> backend dumps last sequence into the dataset folder
    """

    raw_seq = seq_buffer.get_raw_sequence(payload.session_id)
    if raw_seq is None:
        raise HTTPException(
            status_code=400,
            detail="Not enough frames buffered yet. Keep your hand in view until buffer is full.",
        )

    label = payload.label.strip()
    if not label:
        raise HTTPException(status_code=400, detail="Label cannot be empty")

    out_dir = Path(__file__).resolve().parents[4] / "data" / "raw" / label
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time() * 1000)
    path = out_dir / f"sample_{ts}.npz"
    np.savez_compressed(path, landmarks=raw_seq.astype(np.float32), label=label)

    return {"ok": True, "saved": str(path)}
