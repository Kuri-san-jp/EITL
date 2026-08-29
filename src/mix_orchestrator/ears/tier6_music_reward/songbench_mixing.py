"""In-process SongBench (Tencent) **Mixing** scorer.

An in-process version that **follows exactly** the scoring procedure of the
external `external/SongBench/eval.py`, without spawning a subprocess. Written so
that the in-loop reward (min agreement of Audiobox-PQ × SongBench-Mixing) can
keep MuQ and Audiobox loaded together in a single process.

Scoring procedure (must stay consistent with eval.py / model.py):
  1. Load as mono / 24 kHz with librosa.load(..., sr=24000) (an ndarray input is
     likewise downmixed to mono → resampled to 24000 Hz).
  2. Pass audio[None, :] to MuQ "OpenMuQ/MuQ-large-msd-iter" with
     `output_hidden_states=True` → `hidden_states[6]`  (shape [1, T, 1024]).
  3. Run it through Generator(configs/songbench.yaml: in_features=1024,
     ffd_hidden_size=4096, num_classes=7, attn_layer_num=4). The ckpt is
     ckpt/songbench.safetensors via `load_state_dict(..., strict=False)`.
  4. The 7 output dims = [Melody, Arrangement, Musicality, Vocal, Instrumental,
     **Mixing(=index5)**, Structure]. The range comes from tanh*4.5+5.5 inside
     Generator → [1, 10].

Hard policy (memory: feedback_no_proxy_no_experiment): no proxy, no substitute
computation. If MuQ / Generator / ckpt cannot be loaded, raise and stop.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..base import Ear, EarResult


# Output key order of eval.py (the meaning of the Generator's 7 dims). index 5 = Mixing.
_DIMS = ("Melody", "Arrangement", "Musicality", "Vocal",
         "Instrumental", "Mixing", "Structure")
_MIXING_INDEX = 5
_MUQ_HIDDEN_LAYER = 6
_TARGET_SR = 24000


def _default_repo() -> Path:
    """Locate the SongBench repo. Works both inside the container (/mnt) and outside the container."""
    env = os.environ.get("SONGBENCH_REPO")
    if env:
        return Path(env)
    # src/mix_orchestrator/ears/tier6_music_reward/ → project root
    root = Path(__file__).resolve().parents[4]
    cand = root / "external" / "SongBench"
    mnt = Path("/mnt/external/SongBench")
    if mnt.exists() and not cand.exists():
        return mnt
    return cand


class SongBenchMixingEar(Ear):
    """In-process ear returning SongBench's **Mixing** dimension (range [1, 10])."""

    name = "songbench_mixing"
    tier = 6
    cost_per_call_usd = 0.0
    requires_reference = False
    supports_temporal = False
    uses_gpu_model = True

    def __init__(self, device: str = "cuda",
                 repo: Optional[str] = None,
                 muq_model_id: str = "OpenMuQ/MuQ-large-msd-iter"):
        self.repo = Path(repo) if repo else _default_repo()
        self.muq_model_id = muq_model_id
        self._device = device
        self._muq = None
        self._generator = None
        self._torch_device = None

    # ------------------------------------------------------------------
    # model loading (once only)
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._generator is not None and self._muq is not None:
            return
        import torch
        from muq import MuQ
        from omegaconf import OmegaConf
        from safetensors.torch import load_file

        if not self.repo.exists():
            raise FileNotFoundError(
                f"SongBench repo not found: {self.repo} "
                "(can be overridden with SONGBENCH_REPO)")
        cfg_path = self.repo / "configs" / "songbench.yaml"
        ckpt_path = self.repo / "ckpt" / "songbench.safetensors"
        if not cfg_path.exists():
            raise FileNotFoundError(f"config missing: {cfg_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"ckpt missing: {ckpt_path} (the design places it under its academic license)")

        dev = torch.device(self._device) \
            if (self._device != "cuda" or torch.cuda.is_available()) \
            else torch.device("cpu")
        self._torch_device = dev

        # Build the Generator directly (hydra instantiate depends on cwd=repo and
        # on importing model, so we avoid it and read only the yaml values; the
        # structure and initialisation are identical to model.Generator).
        from .._songbench_generator import build_generator
        cfg = OmegaConf.load(str(cfg_path))
        g = cfg.generator
        generator = build_generator(
            in_features=int(g.in_features),
            ffd_hidden_size=int(g.ffd_hidden_size),
            num_classes=int(g.num_classes),
            attn_layer_num=int(g.attn_layer_num),
        ).to(dev).eval()
        state_dict = load_file(str(ckpt_path), device="cpu")
        generator.load_state_dict(state_dict, strict=False)
        self._generator = generator

        muq = MuQ.from_pretrained(self.muq_model_id)
        self._muq = muq.to(dev).eval()

    def release_model(self) -> None:
        self._muq = None
        self._generator = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                                          # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # core scoring
    # ------------------------------------------------------------------
    def _to_mono_24k(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """(C, N) or (N,) → mono / 24 kHz float32, matching eval.py's
        librosa.load(sr=24000). librosa.load defaults to mono=True (channel
        average) and to 'soxr_hq' for resampling, so the downmix/resample of an
        ndarray input goes through librosa as well, to keep them identical."""
        import librosa
        a = audio.mean(axis=0) if audio.ndim == 2 else audio
        a = np.asarray(a, dtype=np.float32)
        if sr != _TARGET_SR:
            a = librosa.resample(a, orig_sr=sr, target_sr=_TARGET_SR)
        return np.ascontiguousarray(a, dtype=np.float32)

    def _score_7dim(self, mono_24k: np.ndarray,
                    chunk_sec: float = 300.0) -> np.ndarray:
        """Compute the 7-dim score with the MuQ model.
        Attention is O(n²), so long songs are split into chunk_sec-second chunks,
        scored, and averaged.

        The env var SB_CHUNK_SEC overrides chunk_sec. A value <= 0 disables
        chunking (= always a single forward over the full song, i.e. the same
        behaviour as before chunking was introduced on . Chunking
        must be disabled for cross-run comparison against the n100 sweeps
        (up to : for songs longer than 300 s the SB value shifts by
        up to ~2.8 (internal notes).
        """
        import torch
        self._ensure_loaded()
        env_cs = os.environ.get("SB_CHUNK_SEC")
        if env_cs is not None:
            chunk_sec = float(env_cs)
        chunk_samples = (int(chunk_sec * _TARGET_SR) if chunk_sec > 0
                         else len(mono_24k) + 1)
        n = len(mono_24k)
        if n <= chunk_samples:
            audio = torch.tensor(mono_24k).unsqueeze(0).to(self._torch_device)
            with torch.no_grad():
                out = self._muq(audio, output_hidden_states=True)
                feat = out["hidden_states"][_MUQ_HIDDEN_LAYER]
                scores = self._generator(feat).squeeze(0)
            return scores.detach().float().cpu().numpy()
        # Long song: split into chunks and average the scores
        chunk_scores = []
        for start in range(0, n, chunk_samples):
            chunk = mono_24k[start:start + chunk_samples]
            audio = torch.tensor(chunk).unsqueeze(0).to(self._torch_device)
            with torch.no_grad():
                out = self._muq(audio, output_hidden_states=True)
                feat = out["hidden_states"][_MUQ_HIDDEN_LAYER]
                scores = self._generator(feat).squeeze(0)
            chunk_scores.append(scores.detach().float().cpu().numpy())
            del audio, out, feat, scores
            torch.cuda.empty_cache()
        return np.mean(chunk_scores, axis=0)

    def score(self, audio: np.ndarray, sr: int) -> float:
        """Return the Mixing score (range [1, 10]). Same as eval.py's `values['Mixing']`."""
        mono = self._to_mono_24k(audio, sr)
        scores = self._score_7dim(mono)
        return float(scores[_MIXING_INDEX])

    def score_all(self, audio: np.ndarray, sr: int) -> dict:
        """Return all 7 dims as a dict (for verification / debugging). Same shape as eval.py's values."""
        mono = self._to_mono_24k(audio, sr)
        scores = self._score_7dim(mono)
        return {dim: round(float(scores[i]), 4) for i, dim in enumerate(_DIMS)}

    # ------------------------------------------------------------------
    # Ear interface
    # ------------------------------------------------------------------
    async def evaluate(self, audio: np.ndarray, sr: int,
                       reference: Optional[np.ndarray] = None,
                       window: Optional[Tuple[float, float]] = None) -> EarResult:
        t0 = time.time()
        if window is not None:
            s, e = window
            audio = audio[:, int(s * sr):int(e * sr)] \
                if audio.ndim == 2 else audio[int(s * sr):int(e * sr)]
        mono = self._to_mono_24k(audio, sr)
        scores = self._score_7dim(mono)
        score_dict = {dim: round(float(scores[i]), 4) for i, dim in enumerate(_DIMS)}
        return EarResult(
            name=self.name,
            score={"Mixing": float(scores[_MIXING_INDEX])},
            elapsed_sec=time.time() - t0,
            raw=score_dict,
        )
