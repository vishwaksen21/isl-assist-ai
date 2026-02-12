from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


@dataclass(frozen=True)
class HandDetectionConfig:
    max_num_hands: int = 1
    model_complexity: int = 1
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6


class MediaPipeHandLandmarker:
    def __init__(self, cfg: HandDetectionConfig = HandDetectionConfig()) -> None:
        self.cfg = cfg
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=cfg.max_num_hands,
            model_complexity=cfg.model_complexity,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )

    def extract_first_hand_landmarks(self, bgr_image: np.ndarray) -> Optional[np.ndarray]:
        """Return landmarks as (21,3) float32 in MediaPipe normalized coords."""
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        res = self._hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None

        hand = res.multi_hand_landmarks[0]
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
        if pts.shape != (21, 3):
            return None
        return pts
