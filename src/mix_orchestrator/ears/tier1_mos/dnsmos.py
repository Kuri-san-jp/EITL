"""DNSMOS wrapper (Microsoft DNS Challenge ONNX baseline)."""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from ..base import Ear, EarResult
from .._proxy_guard import assert_proxy_allowed
from ._proxy_mos import proxy_mos_score


_DNSMOS_SESSION = None     # Module-level cache (CUDA EP + ONNX)


class DNSMOSEar(Ear):
    name = "dnsmos"
    tier = 1
    cost_per_call_usd = 0.0
    uses_gpu_model = True       # Holds the ONNX session on the CUDA EP

    def __init__(self, use_proxy_if_missing: bool = False):
        self.use_proxy = use_proxy_if_missing

    def release_model(self) -> None:
        global _DNSMOS_SESSION
        _DNSMOS_SESSION = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                                          # noqa: BLE001
            pass

    async def evaluate(self, audio: np.ndarray, sr: int,
                       reference: Optional[np.ndarray] = None,
                       window: Optional[Tuple[float, float]] = None) -> EarResult:
        t0 = time.time()
        if window is not None:
            s, e = window
            audio = audio[:, int(s*sr):int(e*sr)] if audio.ndim == 2 else audio[int(s*sr):int(e*sr)]
        try:
            overall = self._real(audio, sr)
            return EarResult(
                name=self.name,
                score={"ovrl_mos": float(overall["ovrl_mos"]),
                       "sig_mos":  float(overall["sig_mos"]),
                       "bak_mos":  float(overall["bak_mos"])},
                elapsed_sec=time.time()-t0,
            )
        except Exception as ex:                                    # noqa: BLE001
            if not self.use_proxy:
                raise
            assert_proxy_allowed(self.name)
            s = proxy_mos_score(audio, sr)
            return EarResult(
                name=self.name,
                score={"ovrl_mos": s, "sig_mos": s, "bak_mos": s},
                elapsed_sec=time.time()-t0,
                warnings=[f"DNSMOS fallback proxy: {ex}"],
            )

    def _real(self, audio, sr) -> dict:
        """DNSMOS P.835 ONNX inference.

        Requires:
          - `onnxruntime` (CPU) or `onnxruntime-gpu` (CUDA EP)
          - the weight file ``sig_bak_ovr.onnx`` (official MS DNS Challenge)
          - the path given via the environment variable ``DNSMOS_ONNX_PATH``
        """
        import os
        path = os.environ.get("DNSMOS_ONNX_PATH")
        if not path:
            raise ImportError(
                "Set the environment variable DNSMOS_ONNX_PATH to the absolute "
                "path of sig_bak_ovr.onnx (docs/install_real_ears.md §3.2)"
            )
        try:
            import onnxruntime as ort
        except ImportError as ex:
            raise ImportError("`pip install onnxruntime-gpu`") from ex

        global _DNSMOS_SESSION
        if _DNSMOS_SESSION is None:
            _DNSMOS_SESSION = ort.InferenceSession(
                path, providers=["CUDAExecutionProvider",
                                 "CPUExecutionProvider"])

        mono = audio.mean(axis=0) if audio.ndim == 2 else audio
        if sr != 16000:
            import librosa
            mono = librosa.resample(mono.astype(np.float32),
                                    orig_sr=sr, target_sr=16000)
        sr = 16000
        # Match the official MS DNSMOS dnsmos_local.py:
        # INPUT_LENGTH = 9.01 seconds (= 144160 samples @ 16 kHz)
        win = int(9.01 * sr)
        if len(mono) < win:
            mono = np.pad(mono, (0, win - len(mono)))
        else:
            start = (len(mono) - win) // 2
            mono = mono[start:start + win]
        inp = mono.astype(np.float32)[None, :]
        out = _DNSMOS_SESSION.run(None, {"input_1": inp})
        # The ONNX output is either [(1, 1), (1, 1), (1, 1)] or [(1, 3)].
        # Following the official dnsmos_local.py, extract the 3 axes [MOS_SIG, MOS_BAK, MOS_OVR].
        arr = np.asarray(out[0]).reshape(-1)
        if arr.size >= 3:
            sig, bak, ovr = float(arr[0]), float(arr[1]), float(arr[2])
        else:
            # For the other shape, pull the scalars out one by one
            sig = float(np.asarray(out[0]).reshape(-1)[0])
            bak = float(np.asarray(out[1]).reshape(-1)[0])
            ovr = float(np.asarray(out[2]).reshape(-1)[0])
        return {"sig_mos": sig, "bak_mos": bak, "ovrl_mos": ovr}
