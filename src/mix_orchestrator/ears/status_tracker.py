"""Read/write ear_real_status.json.

Records, for each ear, the last time it succeeded at evaluating with the real
model. build_ears (experiments/_common.py) reads this JSON and emits a WARNING
for ears that have not been updated for a long time, or that have never been
confirmed.

Usage:

  $ python3 -m mix_orchestrator.ears.status_tracker --refresh
      → initialise every ear for real → evaluate once on a short synthetic
        signal → update the timestamp of the ears that succeeded

  $ python3 -m mix_orchestrator.ears.status_tracker --show
      → list the current status

  # From code:
  from mix_orchestrator.ears.status_tracker import mark_real_success
  mark_real_success("audiobox")
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STATUS_PATH = ROOT / "outputs/ear_real_status.json"


def _load() -> Dict[str, Any]:
    if not STATUS_PATH.exists():
        return {}
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save(status: Dict[str, Any]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")


def mark_real_success(ear_name: str, *, extra: Dict[str, Any] | None = None) -> None:
    """Record that a real (non-proxy) run of the ear succeeded (expected to be
    called from the orchestrator)."""
    status = _load()
    rec = status.get(ear_name, {})
    rec["last_real_success_ts"] = time.time()
    rec["last_real_success_iso"] = datetime.now().isoformat(timespec="seconds")
    if extra:
        rec.update(extra)
    status[ear_name] = rec
    _save(status)


def show() -> None:
    status = _load()
    if not status:
        print(f"{STATUS_PATH} is empty or does not exist."
              " Run `--refresh` to initialise it.")
        return
    now = time.time()
    print(f"{'ear':<25} {'last _real success':<22} {'age':<10}")
    print("-" * 60)
    for name in sorted(status.keys()):
        rec = status[name]
        ts = rec.get("last_real_success_ts", 0)
        iso = rec.get("last_real_success_iso", "—")
        age_days = (now - ts) / 86400 if ts else float("inf")
        age_str = f"{age_days:.1f} d ago" if age_days != float("inf") else "unconfirmed"
        print(f"{name:<25} {iso:<22} {age_str:<10}")


def refresh() -> int:
    """Initialise every ear for real and evaluate once on a short synthetic signal.

    Only the ears that succeed get their status updated; failures are reported
    as messages. The return value is the number of failures (0 means all OK).

    Note: this command runs ears that need a GPU (CLAP/MERT/UTMOS, etc.), so it
    has to be launched through slurm.
    """
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "experiments"))
    import asyncio

    from mix_orchestrator.ears._proxy_guard import forbid_proxy_for_experiment
    forbid_proxy_for_experiment()

    # Candidate ear list (same set as build_ears' expanded mode)
    from mix_orchestrator.ears.tier1_mos.audiobox import AudioboxEar
    from mix_orchestrator.ears.tier1_mos.nisqa import NISQAEar
    from mix_orchestrator.ears.tier1_mos.utmos import UTMOSEar
    from mix_orchestrator.ears.tier1_mos.dnsmos import DNSMOSEar
    from mix_orchestrator.ears.tier1_mos.singmos import SingMOSEar
    from mix_orchestrator.ears.tier1_mos.scoreq import SCOREQEar
    from mix_orchestrator.ears.tier2_reference.clap_similarity import (
        CLAPSimilarityEar,
    )
    from mix_orchestrator.ears.tier2_reference.mert_distance import MERTDistanceEar
    from mix_orchestrator.ears.tier2_reference.visqol_music import ViSQOLMusicEar
    from mix_orchestrator.ears.tier2_reference.cdpam import CDPAMEar
    from mix_orchestrator.ears.tier2_reference.fad import FADEar
    from mix_orchestrator.ears.tier3_judge.qwen2_audio import Qwen2AudioJudge
    from mix_orchestrator.ears.tier5_task.vocal_intelligibility import (
        VocalIntelligibilityEar,
    )
    from mix_orchestrator.ears.tier5_task.genre_classifier import GenreClassifierEar
    from mix_orchestrator.ears.tier5_task.reseparation_sdr import ReseparationSDREar

    rng = np.random.default_rng(0)
    sr = 44100
    audio = rng.normal(0, 0.1, size=(2, sr * 2)).astype(np.float32)
    ref = rng.normal(0, 0.1, size=(2, sr * 2)).astype(np.float32)

    candidates = [
        ("audiobox", AudioboxEar(use_proxy_if_missing=False), None),
        ("nisqa", NISQAEar(use_proxy_if_missing=False), None),
        ("utmos", UTMOSEar(use_proxy_if_missing=False), None),
        ("dnsmos", DNSMOSEar(use_proxy_if_missing=False), None),
        ("singmos", SingMOSEar(use_proxy_if_missing=False), None),
        ("scoreq", SCOREQEar(use_proxy_if_missing=False), None),
        ("clap", CLAPSimilarityEar(use_proxy_if_missing=False), ref),
        ("mert", MERTDistanceEar(use_proxy_if_missing=False), ref),
        # ViSQOL is excluded from the ensemble: hard to build and no wheel
        ("cdpam", CDPAMEar(use_proxy_if_missing=False), ref),
        ("fad", FADEar(use_proxy_if_missing=False), ref),
        ("qwen2_audio_judge", Qwen2AudioJudge(use_proxy_if_missing=False), None),
        ("vocal_intelligibility",
         VocalIntelligibilityEar(use_proxy_if_missing=False), None),
        ("genre", GenreClassifierEar(use_proxy_if_missing=False), None),
        # reseparation_sdr needs attach_stems, hence the special case below
        ("reseparation_sdr",
         ReseparationSDREar(use_proxy_if_missing=False), ref),
    ]

    failed = []
    for name, ear, reference in candidates:
        print(f"[{name}] trying a real run…", flush=True)
        try:
            if name == "reseparation_sdr":
                ear.attach_stems({"vocals": ref, "drums": ref})
            asyncio.run(ear.evaluate(audio, sr, reference=reference))
            mark_real_success(name)
            print(f"  OK: status updated")
        except Exception as ex:                                # noqa: BLE001
            print(f"  FAIL: {type(ex).__name__}: {ex}")
            failed.append(name)
        finally:
            # Free the GPU memory before loading the next ear
            # (policy: never hold two ear models on the GPU at once)
            if getattr(ear, "uses_gpu_model", False):
                try:
                    ear.release_model()
                except Exception:                              # noqa: BLE001
                    pass

    print()
    print(f"succeeded: {len(candidates) - len(failed)} / {len(candidates)}")
    if failed:
        print(f"failed ears: {failed}")
        print("→ see the matching section of docs/install_real_ears.md to fix the environment")
    return len(failed)


def main() -> int:
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--refresh", action="store_true",
                   help="check that every ear runs for real, then update the status")
    g.add_argument("--show", action="store_true",
                   help="show the current status")
    args = parser.parse_args()
    if args.refresh:
        return refresh()
    show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
