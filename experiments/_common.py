"""Shared helpers across experiment runners (E1..E4)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

def _load_env_file() -> None:
    """Apply ROOT/.env to os.environ (existing environment variables are not overwritten).

    python-dotenv lives in python_packages/, but that directory is only put on
    sys.path inside _setup_hf_cache_env() = after this import point, so
    `from dotenv import load_dotenv` raised ImportError inside the container and
    .env was never read (found : ANTHROPIC_API_KEY did not arrive and
    the anthropic backend errored on every song). Read it reliably with a
    minimal dotenv-free parser. Unlike load_dotenv(), which depends on cwd, this
    is anchored at ROOT and therefore independent of the container's cwd.
    """
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    try:
        text = env_path.read_text("utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_env_file()


def _setup_hf_cache_env() -> None:
    """Point the HF / torch / whisper / scoreq caches at the in-project
    hf_cache/ (mounted as /mnt/hf_cache inside the container).

    Same convention as install_real_ears.sh. Export these explicitly in every
    runner process so the experiment runners can load the real ear models even
    in environments without a .env.
    """
    # Absolute path when running outside the container, /mnt when inside the container
    project_cache = ROOT / "hf_cache"
    container_cache = Path("/mnt/hf_cache")
    cache_root = str(container_cache if container_cache.exists()
                     else project_cache)

    os.environ.setdefault("HF_HOME", cache_root)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", f"{cache_root}/hub")
    os.environ.setdefault("TORCH_HOME", f"{cache_root}/torch_cache")
    os.environ.setdefault("XDG_CACHE_HOME", f"{cache_root}/xdg_cache")
    os.environ.setdefault("WHISPER_CACHE_DIR", f"{cache_root}/whisper_cache")
    # Send writes from trust_remote_code modules (GenreClassifierEar etc.) to the writable project cache.
    # the cluster job wrapper forces HF_HOME=/tmp/.cache/huggingface, which makes module writes permission denied,
    # so point only HF_MODULES_CACHE under the project (model reads stay on HF_HOME = Audiobox unchanged).
    os.environ.setdefault("HF_MODULES_CACHE", f"{cache_root}/hf_modules")

    # Persistent pip install target (works around the disposable container).
    # Makes the packages that install_real_ears.sh installed with
    # PIP_TARGET=/mnt/python_packages importable from here.
    container_pkgs = Path("/mnt/python_packages")
    host_pkgs = ROOT / "python_packages"
    pkgs_root = str(container_pkgs if container_pkgs.exists() else host_pkgs)
    if Path(pkgs_root).exists() and pkgs_root not in sys.path:
        sys.path.insert(0, pkgs_root)

    # Add external/SingMOS to PYTHONPATH if it exists
    singmos_root = Path("/mnt/external/SingMOS")
    if not singmos_root.exists():
        singmos_root = ROOT / "external/SingMOS"
    if singmos_root.exists() and str(singmos_root) not in sys.path:
        sys.path.insert(0, str(singmos_root))

    # Add external/NISQA to PYTHONPATH and pass the weight path via env.
    nisqa_root = Path("/mnt/external/NISQA")
    if not nisqa_root.exists():
        nisqa_root = ROOT / "external/NISQA"
    if nisqa_root.exists():
        if str(nisqa_root) not in sys.path:
            sys.path.insert(0, str(nisqa_root))
        weights = nisqa_root / "weights/nisqa.tar"
        if weights.exists():
            os.environ.setdefault("NISQA_WEIGHTS", str(weights))

    # Set the env var if external/dnsmos/sig_bak_ovr.onnx is present.
    dnsmos_onnx = Path("/mnt/external/dnsmos/sig_bak_ovr.onnx")
    if not dnsmos_onnx.exists():
        dnsmos_onnx = ROOT / "external/dnsmos/sig_bak_ovr.onnx"
    if dnsmos_onnx.exists():
        os.environ.setdefault("DNSMOS_ONNX_PATH", str(dnsmos_onnx))

    # SCOREQ downloads its weights to ~/.cache/scoreq/ by design. Pointing HOME
    # under hf_cache avoids the container's /tmp permission problem.
    # Set HOME only in the narrow cases SCOREQ needs, so other tools do not end
    # up depending on it (set it when the original HOME is absent).
    if "HOME" not in os.environ or os.environ["HOME"].startswith("/tmp"):
        os.environ["HOME"] = cache_root

    # MUSDB18-HQ data location (container: /mnt/data/musdb18hq, host: project data/).
    # Making this the in-code default lets the cluster job wrapper be invoked directly without a bash -c wrapper
    # (the experiment runner sets MUSDB18_ROOT itself at import time, so there is no need to inject the env from outside).
    musdb = Path("/mnt/data/musdb18hq")
    if not musdb.exists():
        musdb = ROOT / "data/musdb18hq"
    if musdb.exists():
        os.environ.setdefault("MUSDB18_ROOT", str(musdb))


_setup_hf_cache_env()


def build_ears(mode: str = "default"):
    """Build the ear ensemble used by the experiments.

    Modes:
      - ``"default"``: Tier 1 Audiobox + Tier 4 detectors (5 ears).
        Minimal configuration, for smoke runs.
      - ``"expanded"``: default + the rest of Tier 1 (NISQA / UTMOS / DNSMOS /
        SingMOS / SCOREQ) + all of Tier 2 (CLAP / MERT / ViSQOL / CDPAM /
        FAD) + all of Tier 5 (Vocal Intel / Genre / Reseparation SDR).
        The standard configuration for the main ICASSP experiments (14 ears total).
      - ``"full"``: expanded + the Tier 3 Qwen2-Audio judge. Contains a heavy
        model at ~14 GB VRAM, so use it only for A/B comparisons and final evaluation.

    Hard rule (memory: feedback_no_proxy_no_experiment): substitute computations
    (proxies) are never used in experiments. If the real model cannot be
    imported / loaded, that ear is rejected with an error.

    Enforcement:
      - every ear is constructed with ``use_proxy_if_missing=False``
      - the environment variable ``AUTOMIX_FORBID_PROXY=1`` is exported
        process-wide (``assert_proxy_allowed()`` inside each ear's ``_proxy()``
        then raises)
      - ``outputs/ear_real_status.json`` is read, and a WARNING is emitted for
        any ear whose last successful _real run is more than 7 days old or
        never confirmed
    """
    from mix_orchestrator.ears._proxy_guard import forbid_proxy_for_experiment
    forbid_proxy_for_experiment()

    from mix_orchestrator.ears.tier1_mos.audiobox import AudioboxEar
    from mix_orchestrator.ears.tier4_detector.loudness import LoudnessDetector
    from mix_orchestrator.ears.tier4_detector.true_peak import TruePeakDetector
    from mix_orchestrator.ears.tier4_detector.stereo import StereoCorrelationDetector
    from mix_orchestrator.ears.tier4_detector.lra_dynamics import LRADetector
    base = [
        AudioboxEar(use_proxy_if_missing=False),
        LoudnessDetector(),
        TruePeakDetector(),
        StereoCorrelationDetector(),
        LRADetector(),
    ]

    if mode == "music_noref":
        # No-reference music evaluation set for dry-stem auto-mixing.
        # Excludes the speech-MOS ears (NISQA/UTMOS/DNSMOS/SCOREQ/SingMOS) and
        # the reference-based ones (CLAP/MERT/ViSQOL/CDPAM/FAD), keeping only
        # no-reference metrics that are valid for music. Details:
        # internal notes
        #   - Audiobox 4 axes (no-ref music aesthetic; already in base)
        #   - LUFS / TruePeak / Stereo / LRA (technical metrics; already in base)
        #   - Genre confidence (music task-grounded)
        #   - Re-separation SDR (music task-grounded; dry stems as ground truth)
        from mix_orchestrator.ears.tier5_task import (
            GenreClassifierEar, ReseparationSDREar,
        )
        base.extend([
            GenreClassifierEar(use_proxy_if_missing=False),
            ReseparationSDREar(use_proxy_if_missing=False),
        ])
        _warn_if_ear_status_stale([e.name for e in base])
        return base

    if mode in ("expanded", "full"):
        # Rest of Tier 1
        from mix_orchestrator.ears.tier1_mos.nisqa import NISQAEar
        from mix_orchestrator.ears.tier1_mos.utmos import UTMOSEar
        from mix_orchestrator.ears.tier1_mos.dnsmos import DNSMOSEar
        from mix_orchestrator.ears.tier1_mos.singmos import SingMOSEar
        from mix_orchestrator.ears.tier1_mos.scoreq import SCOREQEar
        # All of Tier 2 (ViSQOL is excluded from the ensemble because its Bazel
        # build is involved and no wheel exists; see docs/install_real_ears.md §3.3)
        from mix_orchestrator.ears.tier2_reference import (
            CLAPSimilarityEar, MERTDistanceEar, CDPAMEar, FADEar,
        )
        # All of Tier 5
        from mix_orchestrator.ears.tier5_task import (
            VocalIntelligibilityEar, GenreClassifierEar, ReseparationSDREar,
        )
        base.extend([
            NISQAEar(use_proxy_if_missing=False),
            UTMOSEar(use_proxy_if_missing=False),
            DNSMOSEar(use_proxy_if_missing=False),
            SingMOSEar(use_proxy_if_missing=False),
            SCOREQEar(use_proxy_if_missing=False),
            CLAPSimilarityEar(use_proxy_if_missing=False),
            MERTDistanceEar(use_proxy_if_missing=False),
            CDPAMEar(use_proxy_if_missing=False),
            FADEar(use_proxy_if_missing=False),
            VocalIntelligibilityEar(use_proxy_if_missing=False),
            GenreClassifierEar(use_proxy_if_missing=False),
            ReseparationSDREar(use_proxy_if_missing=False),
        ])

    if mode == "full":
        # Tier 3 audio-LLM judge (Qwen2-Audio-7B; VRAM ~14 GB)
        from mix_orchestrator.ears.tier3_judge import Qwen2AudioJudge
        base.append(Qwen2AudioJudge(use_proxy_if_missing=False))

    _warn_if_ear_status_stale([e.name for e in base])
    return base


def _warn_if_ear_status_stale(ear_names: List[str], max_age_days: int = 7) -> None:
    """Read outputs/ear_real_status.json and WARN about stale or unconfirmed ears."""
    import json as _json
    import time as _time
    p = ROOT / "outputs/ear_real_status.json"
    if not p.exists():
        print(f"WARNING: outputs/ear_real_status.json does not exist. "
              f"Run `python3 -m mix_orchestrator.ears.status_tracker --refresh` "
              f"to verify that the real ears work.")
        return
    try:
        status = _json.loads(p.read_text("utf-8"))
    except Exception:
        print(f"WARNING: failed to parse ear_real_status.json")
        return
    now = _time.time()
    threshold = max_age_days * 86400
    for name in ear_names:
        rec = status.get(name)
        if rec is None:
            print(f"WARNING: ear={name!r} is not registered in ear_real_status.json — "
                  f"you are about to start an experiment without confirming that it really works")
            continue
        ts = rec.get("last_real_success_ts", 0)
        if (now - ts) > threshold:
            from datetime import datetime as _dt
            last = _dt.fromtimestamp(ts).isoformat() if ts else "none"
            print(f"WARNING: ear={name!r} has had no _real confirmation for over {max_age_days} days "
                  f"(last={last})")


def build_llm(backend: str, tracks: List[str]):
    if backend == "mock":
        from mix_orchestrator.agent.llm_backends.mock_backend import MockBackend
        return MockBackend(tracks=tracks)
    if backend == "local":
        from mix_orchestrator.agent.llm_backends.local_backend import LocalLLMBackend
        # Default arguments are evaluated at module load time, so fetch api_key explicitly here
        api_key = (os.environ.get("LOCAL_LLM_API_KEY") or
                   os.environ.get("OPENAI_API_KEY") or
                   "not-needed")
        # ---- sampling settings (default stays greedy as before / extra params not sent) ----
        # Behaviour changes only when the environment variables are set. Unset means
        # bit-identical results to existing runs. Any change is recorded in
        # llm_stats.sampling so compatibility across runs can be determined.
        def _envf(key: str):
            v = os.environ.get(key)
            return None if v is None or v.strip() == "" else float(v)

        def _envi(key: str):
            v = os.environ.get(key)
            return None if v is None or v.strip() == "" else int(v)

        _temp = _envf("LOCAL_LLM_TEMPERATURE")
        return LocalLLMBackend(
            base_url=os.environ.get("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1"),
            model=os.environ.get("LOCAL_LLM_MODEL", "default"),
            temperature=(0.0 if _temp is None else _temp),
            # Tool calls are short. Keep this low so input(~4k)+output fits in the Qwen server context (8192).
            max_tokens=int(os.environ.get("LOCAL_LLM_MAX_TOKENS", "1536")),
            api_key=api_key,
            top_p=_envf("LOCAL_LLM_TOP_P"),
            top_k=_envi("LOCAL_LLM_TOP_K"),
            seed_base=_envi("LOCAL_LLM_SEED"),
            presence_penalty=_envf("LOCAL_LLM_PRESENCE_PENALTY"),
            frequency_penalty=_envf("LOCAL_LLM_FREQUENCY_PENALTY"),
            repetition_penalty=_envf("LOCAL_LLM_REPETITION_PENALTY"),
        )
    if backend == "gemini":
        from mix_orchestrator.agent.llm_backends.gemini_backend import GeminiBackend
        return GeminiBackend(
            model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            temperature=0.0,
            max_tokens=int(os.environ.get("GEMINI_MAX_TOKENS", "4096")),
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
    from mix_orchestrator.agent.llm_backends.anthropic_backend import AnthropicBackend
    # Set ANTHROPIC_VERTEX_PROJECT_ID when going through Vertex (billing is on the GCP side).
    # Vertex model ids carry a version (e.g. claude-haiku-4-5@20251001), so
    # prefer ANTHROPIC_VERTEX_MODEL when it is present.
    if os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID"):
        model = (os.environ.get("ANTHROPIC_VERTEX_MODEL")
                 or os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5@20251001"))
    else:
        model = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-7")
    return AnthropicBackend(model=model, temperature=0.0)


def load_one(dataset: str, song_id: Optional[str] = None,
             duration_sec: float = 6.0, seed: int = 0,
             split: str = "test"):
    """Load one song's stems, optionally centre-cropped.

    For real datasets (medleydb / musdb18 / cambridge_mt) we crop a
    centred `duration_sec` window so render-and-score is fast in
    smoke tests. Pass `duration_sec <= 0` or `None` to keep full song.
    """
    from mix_orchestrator.data.loaders import load_stems
    from mix_orchestrator.data.song_tier import _center_crop
    if dataset == "synthetic":
        return load_stems("synthetic", duration_sec=duration_sec, seed=seed)
    if not song_id:
        raise ValueError("non-synthetic dataset needs --song / song_id")
    kwargs: Dict[str, Any] = {"song_id": song_id}
    if dataset.lower() in ("musdb18", "musdb18hq", "musdb18-hq", "musdb"):
        # Physically, MUSDB18-HQ has only train/ and test/.
        # "dev" is a logical split defined in splits/dev.json (half of test),
        # so use "test" for the physical path.
        kwargs["split"] = "test" if split == "dev" else split
    stems, sr = load_stems(dataset, **kwargs)
    if duration_sec and duration_sec > 0:
        max_n = int(duration_sec * sr)
        stems = {k: _center_crop(v, max_n) for k, v in stems.items()}
    return stems, sr


def load_split(split: str) -> List[Dict[str, Any]]:
    """Read `outputs/splits/<split>.json` (built by build_splits.py)."""
    import json as _json
    p = ROOT / "outputs/splits" / f"{split}.json"
    if not p.exists():
        return []
    return _json.loads(p.read_text("utf-8"))


def enumerate_split(split: str, tiers: Optional[List[str]] = None,
                    limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return the rows of a built split, optionally filtered by tier.

    Each row has {dataset, song_id, tier, split, rule_fired}.
    """
    rows = load_split(split)
    if tiers:
        rows = [r for r in rows if r.get("tier") in set(tiers)]
    if limit:
        rows = rows[:limit]
    return rows


def enumerate_songs(dataset: str, limit: int = 30,
                    tiers: Optional[List[str]] = None,
                    tier_json: Optional[str] = None,
                    split: str = "test") -> List[Optional[str]]:
    """Return a list of song IDs (or [None] for synthetic, repeated).

    Args:
        tiers: optional ['A', 'B', 'C'] subset to keep
        tier_json: optional path to JSON produced by experiments/classify_songs.py
                   When omitted, looks at outputs/song_tiers/<dataset>.json
    """
    if dataset == "synthetic":
        return [None] * limit
    # For the MUSDB18 logical splits (dev/test), prefer splits/<split>.json when
    # it exists (physically we always read from test/).
    if dataset.lower() in ("musdb18", "musdb18hq", "musdb18-hq", "musdb") \
            and split in ("dev", "test"):
        rows = enumerate_split(split, tiers=tiers, limit=limit)
        if rows:
            return [r["song_id"] for r in rows]
    from mix_orchestrator.data.loaders import list_songs
    list_kwargs = {}
    if dataset.lower() in ("musdb18", "musdb18hq", "musdb18-hq", "musdb"):
        # The physical path is train/ or test/. Map dev to test.
        list_kwargs["split"] = "test" if split == "dev" else split
    try:
        all_songs = list_songs(dataset, **list_kwargs)
    except RuntimeError:
        return [None] * limit

    if tiers:
        from pathlib import Path as _P
        import json as _json
        candidates: List[_P] = []
        if tier_json is not None:
            candidates.append(_P(tier_json))
        # Try split-specific then plain dataset
        candidates.append(ROOT / "outputs/song_tiers" / f"{dataset}_{split}.json")
        candidates.append(ROOT / "outputs/song_tiers" / f"{dataset}.json")
        p = next((c for c in candidates if c.exists()), None)
        if p is None:
            print(f"WARN: tier classification missing under outputs/song_tiers/; "
                  f"run experiments/classify_songs.py first. Returning first "
                  f"{limit} songs without stratification.")
            return all_songs[:limit]
        tier_map = {r["song_id"]: r["tier"] for r in _json.loads(p.read_text("utf-8"))}
        kept = [s for s in all_songs if tier_map.get(s) in set(tiers)]
        return kept[:limit]

    return all_songs[:limit]
