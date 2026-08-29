"""NISQA wrapper (TU Berlin).

Real-mode: requires `pip install nisqa` OR cloning the official repo.
Without it, falls back to the proxy so the pipeline still runs.
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from ..base import Ear, EarResult
from .._proxy_guard import assert_proxy_allowed
from ._proxy_mos import proxy_mos_score


class NISQAEar(Ear):
    name = "nisqa"
    tier = 1
    cost_per_call_usd = 0.0

    def __init__(self, use_proxy_if_missing: bool = False):
        self.use_proxy = use_proxy_if_missing

    async def evaluate(self, audio: np.ndarray, sr: int,
                       reference: Optional[np.ndarray] = None,
                       window: Optional[Tuple[float, float]] = None) -> EarResult:
        t0 = time.time()
        if window is not None:
            s, e = window
            audio = audio[:, int(s*sr):int(e*sr)] if audio.ndim == 2 else audio[int(s*sr):int(e*sr)]
        try:
            score = self._real(audio, sr)
            warnings = []
        except Exception as ex:                                    # noqa: BLE001
            if not self.use_proxy:
                raise
            assert_proxy_allowed(self.name)
            score = proxy_mos_score(audio, sr)
            warnings = [f"NISQA fallback proxy: {ex}"]
        return EarResult(name=self.name, score=float(score),
                         elapsed_sec=time.time()-t0, warnings=warnings)

    def _real(self, audio, sr) -> float:
        """Use the official NISQA model via the `nisqa` package.

        Install: `pip install nisqa` AND clone https://github.com/gabrielmittag/NISQA
        for weights at $NISQA_WEIGHTS/nisqa.tar.
        """
        import os, tempfile, soundfile as sf
        weights = os.environ.get("NISQA_WEIGHTS")
        if not weights:
            raise ImportError("set NISQA_WEIGHTS to checkpoint path")
        try:
            from nisqa.NISQA_model import nisqaModel
        except ImportError as ex:
            raise ImportError("nisqa not installed; `pip install nisqa`") from ex
        # NISQA expects 48kHz mono wav files
        mono = audio.mean(axis=0) if audio.ndim == 2 else audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, mono, sr)
            try:
                args = {"mode": "predict_file", "deg": f.name,
                        "pretrained_model": weights,
                        "ms_channel": None, "tr_bs_val": 1, "tr_num_workers": 0,
                        "output_dir": tempfile.gettempdir(), "csv_deg": None}
                model = nisqaModel(args)
                df = model.predict()
                # df has columns: mos_pred, noi_pred, dis_pred, col_pred, loud_pred
                return float(df.iloc[0]["mos_pred"])
            finally:
                os.unlink(f.name)
