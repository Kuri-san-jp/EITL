"""MEGAMI (Moliner et al. 2025, arXiv:2511.08040) baseline wrapper.

Multitrack Embedding Generative Auto MIxing — a conditional diffusion
generative model from Sony Research. It is permutation-equivariant and accepts
an arbitrary number of stems.

Actual API (external/MEGAMI/inference/inference.py):
  - ``from inference.inference import Inference`` (cwd must be the repo root
    — both the hydra config_path="../conf" and the relative checkpoint path
    "checkpoints/..." depend on cwd).
  - ``Inference(method_args=omegaconf({FxGenerator_code, FxProcessor_code,
    T, Schurn, cfg_scale}))`` loads 2 models + the feature extractor.
  - ``run_inference_single_song(directory=<dir containing the dry .wav files>,
    num_samples, exp_name)`` generates the mix and writes it to
    ``{directory}/{exp_name}/MEGAMI_inference_sample{i}.wav``
    (44.1kHz, PCM_16, stereo).

Constraints:
  - Input dry tracks must be 44.1kHz and at least
    ``load_segment_length=525312`` samples (= 11.9 seconds). Shorter input
    raises ValueError, so call this with duration >= 12s.
  - The 5 weight files must be present in external/MEGAMI/checkpoints/
    (FxGenerator_public.pt / FxProcessor_public.pt / CLAP_DA_public.pt /
    music_audioset_epoch_15_esc_90.14.patched.pt / fxenc_plusplus_default.pt).

References:
  - Paper: https://arxiv.org/abs/2511.08040
  - Code: https://github.com/SonyResearch/MEGAMI
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .external_baseline import ExternalBaseline


_MEGAMI_REPO_ROOT = Path(__file__).resolve().parents[3] / "external/MEGAMI"
_MEGAMI_PKG_DIR = Path(__file__).resolve().parents[3] / "python_packages_megami"

# Checkpoint file names required by the official conf
_REQUIRED_CKPTS = [
    "FxGenerator_public.pt",
    "FxProcessor_public.pt",
    "CLAP_DA_public.pt",
    "music_audioset_epoch_15_esc_90.14.patched.pt",
    "fxenc_plusplus_default.pt",
]


def _ensure_megami_imports() -> None:
    """Prepend the MEGAMI sources and its isolated deps to sys.path."""
    for p in (_MEGAMI_PKG_DIR, _MEGAMI_REPO_ROOT):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


class MegamiBaseline(ExternalBaseline):
    """MEGAMI one-shot generative mixing."""

    name = "megami"
    paper = "Moliner et al. 2025, arXiv:2511.08040"
    requires_reference = False

    # Minimum length MEGAMI requires (samples @ 44.1kHz) = 11.9s
    MIN_SAMPLES_44K = 525312

    def __init__(self,
                 checkpoint_dir: Optional[str] = None,
                 device: str = "cuda",
                 T: int = 100,
                 Schurn: int = 10,
                 cfg_scale: float = 1.0,
                 num_samples: int = 1):
        """
        Args:
            checkpoint_dir: weight directory. Defaults to
                ``external/MEGAMI/checkpoints``.
            device: the actual device is auto-detected as cuda inside MEGAMI.
            T: diffusion sampling steps (official run_inference.sh default 100).
            Schurn: stochastic churn (official default 10).
            cfg_scale: classifier-free guidance scale (official default 1.0).
            num_samples: number of generated samples. 1 for the baseline (one-shot).
        """
        _ensure_megami_imports()
        self.device = device
        self.T = T
        self.Schurn = Schurn
        self.cfg_scale = cfg_scale
        self.num_samples = num_samples
        self.repo_root = _MEGAMI_REPO_ROOT
        self.ckpt_dir = (Path(checkpoint_dir) if checkpoint_dir
                         else _MEGAMI_REPO_ROOT / "checkpoints")

        if not self.ckpt_dir.exists():
            raise FileNotFoundError(
                f"MEGAMI weight directory not found: {self.ckpt_dir}\n"
                "  -> download the weights with ./scripts/download_megami_ckpts.sh."
            )
        missing = [c for c in _REQUIRED_CKPTS if not (self.ckpt_dir / c).exists()]
        if missing:
            raise FileNotFoundError(
                f"MEGAMI checkpoints missing: {missing}\n"
                f"  (check {self.ckpt_dir}; run download_megami_ckpts.sh + "
                "patch_clap_ckpt.py)"
            )

        # Inference is heavy (2 models + CLAP + FxEncoder++), so build it lazily.
        self._inf = None

    # ------------------------------------------------------------------
    def _ensure_inference(self) -> None:
        if self._inf is not None:
            return
        try:
            import omegaconf
        except ImportError as ex:                                # noqa: BLE001
            raise ImportError(
                "omegaconf cannot be imported. "
                "Run `./scripts/install_megami.sh`."
                f"({type(ex).__name__}: {ex})"
            ) from ex

        cwd0 = os.getcwd()
        sys0 = list(sys.path)
        try:
            os.chdir(self.repo_root)
            _ensure_megami_imports()
            from inference.inference import Inference  # type: ignore
            method_args = omegaconf.OmegaConf.create({
                "FxGenerator_code": "public",
                "FxProcessor_code": "public",
                "T": self.T,
                "Schurn": self.Schurn,
                "cfg_scale": self.cfg_scale,
            })
            self._inf = Inference(method_args=method_args)
        except ImportError as ex:                                # noqa: BLE001
            raise ImportError(
                "MEGAMI (inference.inference.Inference) cannot be imported. "
                "Install the dependencies with `./scripts/install_megami.sh`."
                f"({type(ex).__name__}: {ex})"
            ) from ex
        finally:
            os.chdir(cwd0)
            sys.path[:] = sys0

    # ------------------------------------------------------------------
    def _run_samples(self, stems: Dict[str, np.ndarray], sr: int,
                     ) -> Tuple[List[np.ndarray], int]:
        """Run MEGAMI once and return num_samples mixes as (C, L)."""
        import soundfile as sf

        self._ensure_inference()
        main = {k: v for k, v in stems.items() if k != "mixture"}
        if not main:
            raise ValueError("MEGAMI: no dry stems (mixture only?)")
        any_len = next(iter(main.values())).shape[-1]
        min_needed = int(self.MIN_SAMPLES_44K * sr / 44100)
        if any_len < min_needed:
            raise ValueError(
                f"MEGAMI needs at least {min_needed} samples ({min_needed/sr:.1f}s). "
                f"Got {any_len} samples. Call it with a longer duration."
            )

        work = self.repo_root / "tmp_inference" / f"song_{int(time.time()*1000)}"
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)
        mixes: List[np.ndarray] = []
        osr = sr
        try:
            for name, x in main.items():
                arr = np.asarray(x, dtype=np.float32)
                wav = arr.T if arr.ndim == 2 else arr     # (L, C)
                safe = str(name).replace("/", "_").replace(" ", "_")
                sf.write(str(work / f"{safe}.wav"), wav, sr, subtype="FLOAT")

            exp_name = "mixout"
            cwd0 = os.getcwd()
            sys0 = list(sys.path)
            try:
                os.chdir(self.repo_root)
                _ensure_megami_imports()
                self._inf.run_inference_single_song(
                    directory=str(work),
                    num_samples=self.num_samples,
                    exp_name=exp_name,
                )
            finally:
                os.chdir(cwd0)
                sys.path[:] = sys0

            for i in range(self.num_samples):
                wp = work / exp_name / f"MEGAMI_inference_sample{i}.wav"
                if not wp.exists():
                    continue
                m, osr = sf.read(str(wp))                 # (L, 2) or (L,)
                m = np.asarray(m, dtype=np.float32)
                m = m.T if m.ndim == 2 else np.stack([m, m])
                mixes.append(m)
            if not mixes:
                raise FileNotFoundError(
                    f"MEGAMI output not found: {work}/{exp_name}/")
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return mixes, int(osr)

    def mix_samples(self, stems: Dict[str, np.ndarray], sr: int,
                    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Return num_samples generated mixes (so the probe can take best/mean)."""
        t0 = time.time()
        mixes, osr = self._run_samples(stems, sr)
        return mixes, {
            "model_name": "MEGAMI", "elapsed_sec": time.time() - t0,
            "T": self.T, "Schurn": self.Schurn, "cfg_scale": self.cfg_scale,
            "num_samples": len(mixes), "output_sr": osr,
        }

    def mix(self, stems: Dict[str, np.ndarray], sr: int,
            reference: Optional[np.ndarray] = None,
            ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """ExternalBaseline-compatible: return the first generated mix."""
        mixes, meta = self.mix_samples(stems, sr)
        return mixes[0], meta
