from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.routes.predict import seq_buffer


router = APIRouter(tags=["feedback"])


class FeedbackIn(BaseModel):
    session_id: str
    correct_label: str


@router.post("/feedback")
def submit_feedback(payload: FeedbackIn) -> Dict[str, Any]:
    """Store the last full raw landmark sequence for a session under the corrected label.

    This enables an active-learning loop: user corrects output -> new labeled sample gets stored.
    """

    raw_seq = seq_buffer.get_raw_sequence(payload.session_id)
    if raw_seq is None:
        raise HTTPException(status_code=400, detail="No buffered sequence for session")

    out_dir = Path(__file__).resolve().parents[4] / "data" / "raw" / payload.correct_label
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time() * 1000)
    path = out_dir / f"feedback_{ts}.npz"
    np.savez_compressed(path, landmarks=raw_seq.astype(np.float32), label=payload.correct_label)

    return {"ok": True, "saved": str(path)}
