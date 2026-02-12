from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np


@dataclass
class _SessionState:
    frames: Deque[np.ndarray]
    raw_frames: Deque[np.ndarray]
    last_seen: float


class SessionSequenceBuffer:
    def __init__(self, seq_len: int, ttl_seconds: int) -> None:
        self.seq_len = seq_len
        self.ttl_seconds = ttl_seconds
        self._sessions: Dict[str, _SessionState] = {}

    def _gc(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, st in self._sessions.items()
            if now - st.last_seen > self.ttl_seconds
        ]
        for sid in expired:
            self._sessions.pop(sid, None)

    def add_frame(self, session_id: str, keypoints_63: np.ndarray, raw_21x3: np.ndarray) -> int:
        self._gc()
        if session_id not in self._sessions:
            self._sessions[session_id] = _SessionState(
                frames=deque(maxlen=self.seq_len),
                raw_frames=deque(maxlen=self.seq_len),
                last_seen=time.time(),
            )
        st = self._sessions[session_id]
        st.frames.append(keypoints_63.astype(np.float32, copy=False))
        st.raw_frames.append(raw_21x3.astype(np.float32, copy=False))
        st.last_seen = time.time()
        return len(st.frames)

    def get_sequence(self, session_id: str) -> Optional[np.ndarray]:
        self._gc()
        st = self._sessions.get(session_id)
        if not st:
            return None
        if len(st.frames) < self.seq_len:
            return None
        # (seq_len, 63)
        return np.stack(list(st.frames), axis=0).astype(np.float32, copy=False)

    def get_raw_sequence(self, session_id: str) -> Optional[np.ndarray]:
        self._gc()
        st = self._sessions.get(session_id)
        if not st:
            return None
        if len(st.raw_frames) < self.seq_len:
            return None
        # (seq_len, 21, 3)
        return np.stack(list(st.raw_frames), axis=0).astype(np.float32, copy=False)

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
