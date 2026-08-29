"""fxnorm-automix (Martínez-Ramírez et al.) baseline wrapper.

Automix based on effect-normalization plus a Conv-TasNet style regression NN,
proposed in Sony's out-of-domain mixing work (ISMIR2022). Its defining feature is
track normalization via "FXNorm" (shaping of eq / compression / panning /
loudness); it is a one-shot method that regresses the normalized stems into a
single mixture.

This wrapper ports the ``__main__`` block (CLI inference) of
``external/fxnorm-automix/automix/inference.py`` into a form that takes
**in-memory numpy stems** instead of file paths. There is no proxy or fake
anywhere: it runs real inference with the released pretrained weights
``trainings/results/ours_S_Lb`` (the ours_S_Lb configuration used in the paper).

Design:
  - ``__init__``: read the config with ``exec``, ``torch.load`` the ``net``
    (a full nn.Module pickle) -> load the state_dict -> build the ``SuperNet``
    and move it to the GPU. Done only once because it is expensive.
  - ``mix()``: run inference on one song. Apply effect-normalization
    (eq/compression/panning/loudness) sequentially, then regress the mixture
    with SuperNet.inference.

No IRs (impulse responses) ship with the repo, so 'reverb' / 'prereverb' are
removed from ``EFFECTS`` (leaving ['eq','compression','panning','loudness']).
This is exactly the behaviour of the ``inference.py`` CLI when called without IRs.

References:
  - Paper: Martínez-Ramírez et al., "Automatic Music Mixing with
    Deep Learning and Out-of-Domain Data", ISMIR 2022
  - Code: https://github.com/sony/fxnorm-automix
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .external_baseline import ExternalBaseline


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_FXNORM_REPO_ROOT = _PROJECT_ROOT / "external/fxnorm-automix"
_FXNORM_PKG_DIR = _PROJECT_ROOT / "python_packages_fxnorm"

# Weights / config / features of the paper's main configuration (ours_S_Lb)
_DEFAULT_RESULT_DIR = _FXNORM_REPO_ROOT / "trainings/results/ours_S_Lb"
_DEFAULT_NET = _DEFAULT_RESULT_DIR / "net_mixture.dump"
_DEFAULT_WEIGHTS = _DEFAULT_RESULT_DIR / "current_model_for_mixture.params"
_DEFAULT_CONFIG = _FXNORM_REPO_ROOT / "configs/ISMIR/ours_S_Lb.py"
_DEFAULT_FEATURES = _FXNORM_REPO_ROOT / "trainings/features/features_MUSDB18.npy"

# Input stems of MUSDB18 (the names in config['INPUTS'] with '_normalized' stripped).
_STEMS = ["vocals", "bass", "drums", "other"]


def _ensure_fxnorm_imports() -> None:
    for p in (_FXNORM_REPO_ROOT, _FXNORM_PKG_DIR):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


# Cache the inference module, which is built only once.
_FXNORM_INFERENCE_MOD = None

_FIRWIN2_PATCHED = False


def _patch_scipy_firwin2() -> None:
    """Provide compatibility for the ``firwin2(..., nyq=...)`` argument removed in scipy>=1.13.

    ``utils_data_normalization.get_eq_matching`` in fxnorm-automix calls
    ``scipy.signal.firwin2(..., nyq=None, ...)``, but newer scipy removed
    ``nyq`` and replaced it with ``fs``
    (``TypeError: firwin2() got an unexpected keyword argument 'nyq'``).

    The old API's ``nyq`` (Nyquist frequency) is equivalent to the new API's
    ``fs = 2*nyq``, and ``nyq=None`` (= the old default nyq=1.0, freq in [0,1]
    with 1=Nyquist) uses exactly the same frequency convention as the new API's
    ``fs=None`` (= the default fs=2, freq in [0,1] with 1=Nyquist). So this shim
    is a pure compatibility wrapper that does not change behaviour. We do not
    edit the external repo's source; the wrapper absorbs the difference.
    """
    global _FIRWIN2_PATCHED
    if _FIRWIN2_PATCHED:
        return
    import inspect
    import scipy.signal as _ss

    sig = inspect.signature(_ss.firwin2)
    if "nyq" in sig.parameters:
        _FIRWIN2_PATCHED = True
        return  # Old scipy. No patch needed.

    _orig_firwin2 = _ss.firwin2

    def _firwin2_compat(*args, **kwargs):
        if "nyq" in kwargs:
            nyq = kwargs.pop("nyq")
            if nyq is not None and "fs" not in kwargs:
                kwargs["fs"] = 2.0 * nyq
        return _orig_firwin2(*args, **kwargs)

    _ss.firwin2 = _firwin2_compat
    _FIRWIN2_PATCHED = True


_LIBROSA_PATCHED = False


def _patch_librosa_positional() -> None:
    """Restore positional-argument compatibility for APIs that became keyword-only in librosa>=0.10.

    fxnorm-automix assumes an old librosa and calls
      - ``librosa.util.frame(x, window_size, hop_size)``
      - ``librosa.resample(y, orig_sr, target_sr)``
    with **positional arguments**, but in librosa 0.10 ``frame_length`` /
    ``hop_length`` / ``orig_sr`` / ``target_sr`` became keyword-only, so it dies
    with e.g. ``TypeError: frame() takes 1 positional argument but 3 were
    given``. This is a pure positional -> keyword conversion shim and does not
    change behaviour. We do not edit the external repo; the wrapper absorbs it.
    """
    global _LIBROSA_PATCHED
    if _LIBROSA_PATCHED:
        return
    import librosa

    _orig_frame = librosa.util.frame

    def _frame_compat(x, *args, **kwargs):
        if args:
            names = ["frame_length", "hop_length", "axis"]
            for name, val in zip(names, args):
                kwargs.setdefault(name, val)
        return _orig_frame(x, **kwargs)

    librosa.util.frame = _frame_compat

    _orig_resample = librosa.resample

    def _resample_compat(y, *args, **kwargs):
        if args:
            names = ["orig_sr", "target_sr"]
            for name, val in zip(names, args):
                kwargs.setdefault(name, val)
        return _orig_resample(y, **kwargs)

    librosa.resample = _resample_compat
    _LIBROSA_PATCHED = True


def _import_fxnorm_inference():
    """Import ``automix.inference`` without the bug inherited from its ``__main__`` block.

    ``external/fxnorm-automix/automix/inference.py`` has a dedent bug at the end
    (L615-620): module top-level code references ``start_time``, which is only
    defined inside ``__main__``. A plain ``import automix.inference`` therefore
    always dies with ``NameError: name 'start_time' is not defined``.

    Our policy is not to edit the external repo's source (it would break other
    projects and be lost on a re-clone), so we read the source, truncate
    everything from ``if __name__ == '__main__':`` onwards, and exec it as a
    module named ``automix.inference``. This safely gives us the module-level
    constants (EFFECTS / MAX_LENGTH / COMPUTE_NORMALIZATION / CPU_COUNT) and
    functions (normalize_audio_wave / smooth_feature) without the side effects
    (argparse / file I/O / the trailing bug).
    """
    global _FXNORM_INFERENCE_MOD
    if _FXNORM_INFERENCE_MOD is not None:
        return _FXNORM_INFERENCE_MOD

    _ensure_fxnorm_imports()
    _patch_scipy_firwin2()
    _patch_librosa_positional()
    import types

    src_path = _FXNORM_REPO_ROOT / "automix/inference.py"
    src = src_path.read_text()
    # Strip everything from the `__main__` block onwards (= CLI-only code plus
    # the trailing dedent bug).
    marker = "if __name__ == '__main__':"
    idx = src.find(marker)
    if idx != -1:
        src = src[:idx]

    mod = types.ModuleType("automix.inference")
    mod.__file__ = str(src_path)
    mod.__package__ = "automix"
    # The automix package itself imports normally (no side effects), so register
    # the truncated module in sys.modules to satisfy relative imports etc.
    sys.modules["automix.inference"] = mod
    code = compile(src, str(src_path), "exec")
    exec(code, mod.__dict__)
    _FXNORM_INFERENCE_MOD = mod
    return mod


def _to_stereo_NC(arr: np.ndarray, n_channels: int = 2) -> np.ndarray:
    """Reshape a stem of arbitrary shape into (N, C=n_channels) float32.

    The input is expected to be (C, N) per the MixState convention, or (N,).
    FxNorm inference internally requires (N, C), so we transpose.
    """
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]                              # (N,) -> (N, 1)
    elif a.ndim == 2:
        # Callers pass (C, N). C is in {1,2}, so treat the shorter axis as channels.
        if a.shape[0] <= 2 and a.shape[0] < a.shape[1]:
            a = a.T                                 # (C, N) -> (N, C)
        # Otherwise (already (N, C)) leave it as is.
    else:
        raise ValueError(f"unexpected stem ndim={a.ndim}, shape={a.shape}")

    c = a.shape[1]
    if c == n_channels:
        return np.ascontiguousarray(a, dtype=np.float32)
    if c == 1 and n_channels == 2:
        return np.ascontiguousarray(np.repeat(a, 2, axis=1), dtype=np.float32)
    if c == 2 and n_channels == 1:
        return np.ascontiguousarray(a.mean(axis=1, keepdims=True), dtype=np.float32)
    raise ValueError(f"cannot map {c} channels to {n_channels}")


class FXNormBaseline(ExternalBaseline):
    """fxnorm-automix one-shot regression mixing (ISMIR2022)."""

    name = "fxnorm"
    paper = "Martínez-Ramírez et al., ISMIR 2022"
    requires_reference = False

    def __init__(self,
                 checkpoint_path: Optional[str] = None,
                 device: str = "cuda",
                 net_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 features_path: Optional[str] = None):
        """
        Args:
            checkpoint_path: state_dict (``.params``). Defaults to
                ``trainings/results/ours_S_Lb/current_model_for_mixture.params``.
            device: "cuda" or "cpu".
            net_path: full nn.Module pickle (``net_mixture.dump``).
            config_path: config (``configs/ISMIR/ours_S_Lb.py``).
            features_path: effect-normalization features
                (``features_MUSDB18.npy``).
        """
        _ensure_fxnorm_imports()
        self.device = device
        self.net_path = Path(net_path) if net_path else _DEFAULT_NET
        self.weights_path = (Path(checkpoint_path) if checkpoint_path
                             else _DEFAULT_WEIGHTS)
        self.config_path = Path(config_path) if config_path else _DEFAULT_CONFIG
        self.features_path = (Path(features_path) if features_path
                              else _DEFAULT_FEATURES)

        for p, label in [(self.net_path, "net.dump"),
                         (self.weights_path, "weights(.params)"),
                         (self.config_path, "config(.py)"),
                         (self.features_path, "features(.npy)")]:
            if not p.exists():
                raise FileNotFoundError(
                    f"fxnorm-automix {label} not found: {p}\n"
                    "  -> check the external/fxnorm-automix clone and the weight placement."
                )

        self._build_model()
        self._load_features()

    # ------------------------------------------------------------------
    def _build_model(self) -> None:
        """Load the config and build the SuperNet (ported from inference.py L394-432)."""
        import torch  # noqa: F401  (torch 2.x in the image)
        from automix.common_supernet import SuperNet

        # --- load the config with exec (same as the CLI) ---
        # exec(open(config).read()) creates the config dict in the current namespace.
        ns: Dict[str, Any] = {}
        with open(self.config_path) as f:
            exec(compile(f.read(), str(self.config_path), "exec"), ns)
        config = ns["config"]
        self.config = config

        self.n_channels = config["N_CHANNELS"]                 # 2
        self.accepted_sr = config["ACCEPTED_SAMPLING_RATES"]   # [44100]
        self.SR = min(self.accepted_sr)                        # 44100
        # STEMS = the names in config['INPUTS'] with '_normalized' stripped.
        self.STEMS = [i.split("_")[0] for i in config["INPUTS"]]
        assert self.STEMS == _STEMS, (self.STEMS, _STEMS)

        # --- load the net (full nn.Module pickle, saved with torch 1.9) ---
        # The torch in the image is 2.x, where the default of weights_only
        # changed to True, so weights_only=False is required to unpickle a full
        # module.
        try:
            net = torch.load(self.net_path, map_location="cpu",
                             weights_only=False)
        except TypeError:
            # Older torch has no weights_only argument.
            net = torch.load(self.net_path, map_location="cpu")
        net.load_state_dict(
            torch.load(self.weights_path,
                       map_location=lambda storage, loc: storage))

        # --- unfolding_params (None when BATCHED_TEST=False) ---
        unfolding_params = None
        if config["BATCHED_TEST"]:
            unfolding_params = {
                "window_size": config["TRAINING_SEQ_LENGTH"],
                "guard_left": config["GUARD_LEFT"],
                "guard_right": config["GUARD_RIGHT"],
                "input_type": net.input_type,
            }

        super_net = SuperNet(
            net,
            stft_window=torch.from_numpy(
                config["STFT_WINDOW"].astype(np.float32)),
            stft_hop_length=config["HOP_LENGTH"],
            batched_valid=config["BATCHED_TEST"],
            unfolding_params=unfolding_params,
            training_length=config["TRAINING_SEQ_LENGTH"],
            training_batch_size=config["BATCH_SIZE"] // 1,
            use_amp=config["USE_AMP"],
        )
        super_net.to(self.device)
        super_net.eval()
        if config["QUANTIZATION_OP"] is not None:
            super_net.quantize(config["QUANTIZATION_OP"],
                               config["QUANTIZATION_BW"])

        self.super_net = super_net
        self.kernel_size_encoder = config["KERNEL_SIZE_ENCODER"]
        self.outputs = config["OUTPUTS"]                       # ['mixture']

        # --- windowed (unfolded) inference params (to stay under the cuDNN limits) ---
        # ours_S_Lb has BATCHED_TEST=False, so super_net itself forwards the long
        # sequence in one go (= CUDNN_STATUS_NOT_SUPPORTED from cuDNN at 240s).
        # The wrapper always applies the same mechanism the CLI uses with
        # BATCHED_TEST=True (unfold/reconstruct_from_unfold), keeping each forward
        # at training_length (~10s). window_size = TRAINING_SEQ_LENGTH
        # (~440960 = 10s) and guard = GUARD_LEFT/RIGHT (~32446 = 0.74s) are the
        # same as during training, so boundary artifacts match training.
        self._unfold_params = {
            "window_size": int(config["TRAINING_SEQ_LENGTH"]),
            "guard_left": int(config["GUARD_LEFT"]),
            "guard_right": int(config["GUARD_RIGHT"]),
            "input_type": net.input_type,
        }
        # Number of windows per forward (kept small on the safe side for
        # cuDNN/VRAM). BATCH_SIZE is 4 during training. For long stems n_windows
        # reaches several dozen, so we split the forward into training_batch_size
        # chunks.
        self._infer_batch_size = max(1, int(config["BATCH_SIZE"]))

    # ------------------------------------------------------------------
    def _load_features(self) -> None:
        """Load and smooth the effect-normalization features (inference.py L561-564).

        ``smooth_feature`` references the module globals ``STEMS`` / ``EFFECTS``.
        ``EFFECTS`` exists at module level, but ``STEMS`` is only defined inside
        __main__, so we inject it into the inference module's namespace here.
        """
        fx_inf = _import_fxnorm_inference()

        # The module-level code in inference.py only defines ``SR`` / ``STEMS``
        # inside __main__, yet smooth_feature references ``STEMS`` and
        # normalize_audio_wave references ``SR`` as module globals. Inject them
        # into the truncated module.
        fx_inf.STEMS = list(self.STEMS)
        fx_inf.SR = self.SR

        features_mean = np.load(self.features_path, allow_pickle=True)[()]
        # smooth_feature modifies the dict in place and returns the same dict.
        self.features_mean = fx_inf.smooth_feature(features_mean)

    # ------------------------------------------------------------------
    def _normalize_stems(self, data: np.ndarray) -> np.ndarray:
        """Apply effect-normalization to each stem sequentially (inference.py L561-574).

        Args:
            data: shape (S=4, 1, T, C) float32.
        Returns:
            An array of the same shape after normalization.
        """
        fx_inf = _import_fxnorm_inference()

        # No IRs, so drop reverb / prereverb. EFFECTS is a module-level list, so
        # **copy it first** and then remove, to avoid destroying the original.
        effects: List[str] = list(fx_inf.EFFECTS)
        for fx in ("prereverb", "reverb"):
            if fx in effects:
                effects.remove(fx)
        # Just in case: re-inject the module globals referenced by
        # normalize_audio_wave / smooth_feature.
        fx_inf.STEMS = list(self.STEMS)
        fx_inf.SR = self.SR

        if not fx_inf.COMPUTE_NORMALIZATION:
            return data

        # multiprocessing.Pool is hard to debug, so call it in a sequential loop.
        for k, inp in enumerate(self.STEMS):
            out = fx_inf.normalize_audio_wave(
                (data[k][0], effects, inp, self.features_mean))
            data[k][0] = out
        return data

    # ------------------------------------------------------------------
    def _windowed_inference(self, test_data: "Any") -> "Any":
        """Unfold ``SuperNet.inference`` so long inputs stay under the cuDNN limits.

        ``ours_S_Lb`` has ``BATCHED_TEST=False``, so ``super_net.inference``
        forwards the full length (240s ~ 10.6M samples) in one go and Conv1d
        exceeds the cuDNN limits (``CUDNN_STATUS_NOT_SUPPORTED`` / the 2^31
        element and workspace constraints) and dies. This method applies, on the
        wrapper side and without changing ``batched_valid``, **exactly the same
        mechanism** the CLI uses with ``BATCHED_TEST=True``: ``unfold`` /
        ``reconstruct_from_unfold``.

        Procedure (TIME_SAMPLES only; ours_S_Lb has input_type=TIME_SAMPLES):
          1. Equivalent of preprocess: permute ``(1+S, B=1, T, C)`` to
             ``(1+S, 1, C, T)``.
          2. Split each stem with ``unfold`` into ``(n_windows, C, window_size)``
             (overlapping by the guard; window_size = training_length ~ 10s).
          3. ``hstack`` the stems into ``(n_windows, C*S, window_size)``, forward
             it through the net ``training_batch_size`` at a time, and concat
             along the time axis.
          4. With ``reconstruct_from_unfold``, discard the guard of each window,
             concatenate the central ``hop = window_size - guard_left -
             guard_right``, and restore the original ``original_length``.

        Discarding the guard regions (~0.74s on each side) keeps the boundary
        behaviour and the continuity of the LSTM state identical to training
        (validation with BATCHED_TEST).

        Args:
            test_data: shape ``(1+S, 1, T, C)`` float32 tensor (GPU).
                S=4, T=new_samples (already kernel-padded), C=2.
        Returns:
            A dict ``{DataType.TIME_SAMPLES: (n_targets, 1, C, T)}``.
        """
        import torch
        from automix.common_datatypes import (
            DataType, unfold, reconstruct_from_unfold, get_length,
        )

        net = self.super_net.net
        if net.input_type != DataType.TIME_SAMPLES:
            # input_types other than ours_S_Lb's are outside this wrapper's scope.
            raise ValueError(
                f"_windowed_inference is TIME_SAMPLES only "
                f"(net.input_type={net.input_type}).")
        n_stems = net.n_stems
        up = self._unfold_params

        # LSTM/GRU assume multi-GPU and need flattening (same as preprocess).
        for m in net.children():
            if isinstance(m, (torch.nn.LSTM, torch.nn.GRU)):
                m.flatten_parameters()

        with torch.cuda.amp.autocast(enabled=self.super_net.use_amp):
            # (1+S, 1, T, C) -> (1+S, 1, C, T). Put the time axis last.
            x = test_data.permute((0, 1, 3, 2)).contiguous()
            original_length = get_length(x[0], DataType.TIME_SAMPLES)

            # Unfold each track (mixture target + S stems). x[i] is (1, C, T).
            # unfold -> (n_windows, C, window_size).
            x = torch.stack([unfold(_x, **up) for _x in x.unbind(0)], dim=0)

            # hstack the stem side (the last n_stems entries): (n_windows, C*S, window_size).
            inp = torch.hstack(torch.unbind(x[-n_stems:]))

            # Split n_windows into training_batch_size chunks and forward each.
            # Each forward is (<=bs, C*S, window_size) with window_size ~ 10s
            # -> cuDNN is fine.
            outs = [net(_chunk)
                    for _chunk in torch.split(inp, self._infer_batch_size)]
            out_unfolded = torch.cat(outs, dim=1)  # (n_targets, n_windows, C, win)

            # Discard the guard and restore the original length: (n_targets, 1, C, original_length).
            out_time = reconstruct_from_unfold(
                out_unfolded, original_length=original_length, **up)

        return {DataType.TIME_SAMPLES: out_time}

    # ------------------------------------------------------------------
    def mix(self,
            stems: Dict[str, np.ndarray],
            sr: int,
            reference: Optional[np.ndarray] = None,
            ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """One-shot inference. stems (C, N) -> mixture (2, N).

        In-memory version of inference.py __main__ (L443-603).
        """
        import torch
        from automix.common_datatypes import DataType
        fx_inf = _import_fxnorm_inference()

        t0 = time.time()
        if sr != self.SR:
            raise ValueError(
                f"FxNorm expects sr={self.SR}; got input sr={sr}. "
                "The caller must resample to 44100.")

        # --- reshape stems to (N, C) float32 [-1,1] and move to int16-scale ---
        # In the CLI, load_wav returns int16 and it is assigned straight into a
        # float32 zeros array (= int16-scale). This wrapper's input is [-1,1]
        # float, so multiply by (1 + iinfo(int16).max) to match that scale. The
        # loudness stage at the end of effect-normalization always renormalizes
        # to [-1,1], so the scale difference does not affect the output, but we
        # follow this convention to match the CLI exactly.
        int16_max = np.iinfo(np.int16).max
        prepared: Dict[str, np.ndarray] = {}
        for name in self.STEMS:
            if name in stems and stems[name] is not None:
                wav_nc = _to_stereo_NC(stems[name], self.n_channels)
                prepared[name] = wav_nc * (1.0 + int16_max)
            else:
                prepared[name] = None  # missing stems become zeros later

        # Determine max_samples (the longest present stem, capped by MAX_LENGTH).
        present_lens = [v.shape[0] for v in prepared.values() if v is not None]
        if not present_lens:
            raise ValueError("FxNorm: all input stems are missing.")
        max_samples = min(max(present_lens), int(fx_inf.MAX_LENGTH * self.SR))

        # data: (S, 1, T, C) float32
        data = np.zeros((len(self.STEMS), 1, max_samples, self.n_channels),
                        dtype=np.float32)
        for k, name in enumerate(self.STEMS):
            v = prepared[name]
            if v is None:
                continue                                    # leave as zeros
            seg = v[:max_samples]
            data[k][0][:seg.shape[0]] = seg                 # shorter stems are zero-padded

        # Pad the tail to match the encoder kernel (inference.py L532-533).
        new_samples = (1 + max_samples // self.kernel_size_encoder) \
            * self.kernel_size_encoder
        data = np.pad(data,
                      [(0, 0), (0, 0), (0, new_samples - max_samples), (0, 0)])

        # --- effect-normalization (eq / compression / panning / loudness) ---
        data = self._normalize_stems(data)

        # --- forward (windowed SuperNet.inference) ---
        # A single full-length forward dies by exceeding the cuDNN limits, so we
        # run windowed inference with the same mechanism as the CLI's
        # unfold/reconstruct_from_unfold (see _windowed_inference).
        # effect-normalization was already applied once over the full length in
        # _normalize_stems above; only the forward is windowed.
        test_data = torch.from_numpy(data).to(self.device)
        with torch.no_grad():
            test_out = self._windowed_inference(test_data)
            audio_out = test_out[DataType.TIME_SAMPLES].cpu().numpy()
            audio_out = audio_out[..., :max_samples]
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        # k=0 of OUTPUTS=['mixture']. audio_out shape =
        # (n_targets=1, batch=1, n_channels=2, n_samples).
        # So audio_out[0, 0] = (C, N).
        mix = audio_out[0, 0]
        mix = np.asarray(mix, dtype=np.float32)
        # Guarantee (C, N). Transpose in the unlikely case it is (N, C).
        if mix.ndim == 1:
            mix = np.stack([mix, mix])
        elif mix.shape[0] != self.n_channels and mix.shape[-1] == self.n_channels:
            mix = mix.T
        if mix.shape[0] == 1 and self.n_channels == 2:
            mix = np.repeat(mix, 2, axis=0)
        mix = np.ascontiguousarray(mix, dtype=np.float32)

        return mix, {
            "model_name": "fxnorm-automix",
            "config": "ours_S_Lb",
            "elapsed_sec": time.time() - t0,
            "device": self.device,
            "sr": self.SR,
            "effects": ["eq", "compression", "panning", "loudness"],
            "ir_used": False,
            "out_shape": list(mix.shape),
            "windowed_inference": True,
            "window_size": self._unfold_params["window_size"],
            "guard": self._unfold_params["guard_left"],
        }
