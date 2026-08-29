"""REAPER-backed renderer matching the `render(state) -> np.ndarray` contract.

Phase 2 only — use the pedalboard renderer for Phase 1 reported results.
This file lets us A/B identical-output between back-ends before swapping.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from ..dsp.mix_state import MixState
from ..dsp.effects import (
    StaticGain, StaticPan, StaticWidth, StaticEQ, StaticCompressor,
    MasterEQ, MasterCompressor, MasterLimiter,
)
from .reaper_mcp_bridge import ReaperMCPBridge


REAPER_FX_NAME = {
    StaticEQ:         "ReaEQ",
    StaticCompressor: "ReaComp",
    MasterEQ:         "ReaEQ",
    MasterCompressor: "ReaComp",
    MasterLimiter:    "ReaLimit",
}


class ReaperRenderer:
    """Same call signature as `dsp.renderer.render` so the orchestrator
    can be re-targeted by swapping a single import."""

    def __init__(self, bridge: Optional[ReaperMCPBridge] = None,
                 work_dir: Optional[str] = None):
        self.bridge = bridge or ReaperMCPBridge()
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="reaper_render_")

    def __call__(self, state: MixState) -> np.ndarray:
        return self.render(state)

    def render(self, state: MixState) -> np.ndarray:
        if not self.bridge._connected:
            self.bridge.connect()
        self.bridge.reset_session()
        sr = state.sample_rate
        duration = max(s.shape[-1] for s in state.stems.values()) / sr

        # Write each stem to a temp wav and import as a separate REAPER track
        for trk_name, stem in state.stems.items():
            wav_path = Path(self.work_dir) / f"{trk_name}.wav"
            arr = stem.T if stem.ndim == 2 else stem
            sf.write(str(wav_path), arr, sr)
            idx = self.bridge.add_track(trk_name)
            self.bridge.import_audio(idx, str(wav_path), start_sec=0.0)
            # Apply per-stem static effects
            for eff in state.static_processors.get(trk_name, ()):
                self._apply_effect_to_track(idx, eff)

        # Master chain
        for eff in state.master_chain:
            self._apply_effect_to_master(eff)

        # Render
        out_path = Path(self.work_dir) / "out.wav"
        audio = self.bridge.render_master(0.0, duration, sr, str(out_path))
        return audio.astype(np.float32)

    # ---------- effect translation ----------

    def _apply_effect_to_track(self, track_idx: int, eff) -> None:
        if isinstance(eff, StaticGain):
            self.bridge.set_track_volume(track_idx, eff.gain_db)
            return
        if isinstance(eff, StaticPan):
            self.bridge.set_track_pan(track_idx, eff.pan)
            return
        fx_name = REAPER_FX_NAME.get(type(eff))
        if not fx_name:
            return
        params = self._effect_params(eff)
        self.bridge.add_fx(track_idx, fx_name, params)

    def _apply_effect_to_master(self, eff) -> None:
        # The bonfire MCP exposes the master track at idx -1 (convention).
        fx_name = REAPER_FX_NAME.get(type(eff))
        if not fx_name:
            return
        self.bridge.add_fx(track_idx=-1, fx_name=fx_name, params=self._effect_params(eff))

    @staticmethod
    def _effect_params(eff) -> dict:
        # The exact param keys depend on the FX. The mapping here is intentionally
        # generic; Phase 2 will reconcile names with the actual ReaEQ/ReaComp slots.
        if isinstance(eff, (StaticEQ, MasterEQ)):
            return {"band1_freq_hz": eff.freq,
                    "band1_gain_db": eff.gain_db,
                    "band1_q":       eff.q}
        if isinstance(eff, (StaticCompressor, MasterCompressor)):
            return {"threshold_db": eff.threshold_db,
                    "ratio":        eff.ratio,
                    "attack_ms":    eff.attack_ms,
                    "release_ms":   eff.release_ms}
        if isinstance(eff, MasterLimiter):
            return {"ceiling_db": eff.ceiling_db, "release_ms": eff.release_ms}
        return {}
