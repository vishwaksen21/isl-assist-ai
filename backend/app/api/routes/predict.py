from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.ml.inference import SignPredictor
from app.ml.preprocess import normalize_and_flatten_frame
from app.services.mediapipe_service import MediaPipeHandLandmarker
from app.services.session_buffer import SessionSequenceBuffer


router = APIRouter(tags=["predict"])


_ARTIFACTS_DIR = Path(__file__).resolve().parents[4] / "artifacts"
_MODEL_PATH = _ARTIFACTS_DIR / "model.pt"
_LABELS_PATH = _ARTIFACTS_DIR / "labels.json"


hand_landmarker = MediaPipeHandLandmarker()
seq_buffer = SessionSequenceBuffer(
    seq_len=settings.seq_len,
    ttl_seconds=settings.session_ttl_seconds,
)

_predictor: Optional[SignPredictor] = None


def _get_predictor() -> SignPredictor:
    global _predictor
    if _predictor is None:
        if not _MODEL_PATH.exists() or not _LABELS_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model artifacts not found. Train first to create "
                    f"{_MODEL_PATH} and {_LABELS_PATH}."
                ),
            )
        _predictor = SignPredictor(_MODEL_PATH, _LABELS_PATH, device="cpu")
    return _predictor


@router.get("/session")
def new_session() -> Dict[str, str]:
    return {"session_id": str(uuid.uuid4())}


@router.post("/predict/frame")
async def predict_from_frame(
    frame: UploadFile = File(...),
    session_id: str = Form(...),
) -> Dict[str, Any]:
    data = await frame.read()
    img_np = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    landmarks = hand_landmarker.extract_first_hand_landmarks(bgr)
    if landmarks is None:
        return {
            "detected": False,
            "session_id": session_id,
            "buffer_size": 0,
        }

    keypoints_63 = normalize_and_flatten_frame(landmarks)
    buf_size = seq_buffer.add_frame(session_id, keypoints_63, landmarks)

    seq = seq_buffer.get_sequence(session_id)
    if seq is None:
        return {
            "detected": True,
            "session_id": session_id,
            "buffer_size": buf_size,
            "ready": False,
        }

    predictor = _get_predictor()
    label, conf, dist = predictor.predict(seq)

    return {
        "detected": True,
        "ready": True,
        "session_id": session_id,
        "buffer_size": buf_size,
        "prediction": label,
        "confidence": conf,
        "distribution": dist,
    }


@router.post("/session/reset")
def reset_session(session_id: str) -> Dict[str, Any]:
    seq_buffer.reset(session_id)
    return {"ok": True, "session_id": session_id}
