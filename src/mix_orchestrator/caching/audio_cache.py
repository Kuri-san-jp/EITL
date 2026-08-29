"""Disk cache for rendered audio (keyed by state hash)."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


class AudioCache:
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[np.ndarray]:
        p = self.root / f"{key}.wav"
        if not p.exists():
            return None
        data, _sr = sf.read(str(p), dtype="float32", always_2d=True)
        return data.T

    def put(self, key: str, audio: np.ndarray, sr: int) -> None:
        p = self.root / f"{key}.wav"
        arr = audio.T if audio.ndim == 2 else audio
        sf.write(str(p), arr, sr)

    @staticmethod
    def hash(*parts: str) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(p.encode())
        return h.hexdigest()[:16]
