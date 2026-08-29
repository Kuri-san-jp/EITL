"""Evaluate each method with KAD/FAD (the distributional metrics from MEGAMI).
The set of saved mixes vs the distribution of MUSDB professional mixes.
KAD=Gaussian-kernel MMD, FAD=Frechet, in the CLAP embedding space. Lower means
closer to the professional distribution = better.
Note: distributional metrics need many samples. 6 songs is a small sample and
unstable. Uses the GPU (CLAP). Run via slurm.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from _common import load_one, enumerate_split  # noqa: E402


def _clap_embed_fn():
    import sys as _s, os
    meg = str(ROOT / "external/MEGAMI")
    if meg not in _s.path:
        _s.path.insert(0, meg)
    cwd0 = os.getcwd()
    try:
        os.chdir(meg)
        from utils.laion_clap.hook import CLAP_Module
        import torch, torchaudio
        model = CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        model.load_ckpt("checkpoints/music_audioset_epoch_15_esc_90.14.patched.pt")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(dev)

        def embed(audio, sr):
            mono = audio.mean(axis=0) if audio.ndim == 2 else audio
            x = torch.from_numpy(np.asarray(mono, np.float32)).unsqueeze(0)
            x = torchaudio.functional.resample(x, sr, 48000)
            with torch.no_grad():
                e = model.get_audio_embedding_from_data(x.to(dev), use_tensor=True)
            return e.cpu().numpy().reshape(-1)
        return embed
    finally:
        os.chdir(cwd0)


def _mmd_gaussian(X, Y, sigma=None):
    # NOTE: the kernel bandwidth sigma is estimated from the reference (Y) only
    # (median heuristic on Y). Estimating it on vstack([X,Y]) makes sigma move
    # with the sample size and distribution of X, which makes absolute KAD values
    # incomparable across methods with different n (review fix stats-major).
    # Fixing it to Y = the reference means every method is evaluated at the same
    # kernel scale, and it stays consistent under bootstrap. Calls are always in
    # the order _mmd_gaussian(method_emb, reference_emb) (2nd argument = reference).
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)

    def k(a, b, s):
        aa = (a**2).sum(1)[:, None]; bb = (b**2).sum(1)[None, :]
        d2 = np.maximum(aa + bb - 2 * a @ b.T, 0)
        return np.exp(-d2 / (2 * s * s))
    if sigma is None:
        # Median pairwise euclidean distance of Y in pure numpy (no scipy.pdist;
        # this avoids the broken vendored scipy outside the container and lets the CPU
        # summarize run directly outside the container).
        gy = Y @ Y.T
        dy = np.diag(gy)
        d2 = np.maximum(dy[:, None] + dy[None, :] - 2 * gy, 0)
        iu = np.triu_indices(len(Y), k=1)
        sigma = float(np.median(np.sqrt(d2[iu]))) + 1e-9
    # Unbiased MMD^2 U-statistic (within-set, the diagonal = self-pairs is
    # excluded). With the biased V-statistic, duplicate pairs in a bootstrap
    # resample inflate the diagonal term and bias KAD upward, pushing the point
    # estimate outside the CI. The KAD (Kernel Audio Distance) literature also
    # uses the unbiased estimator. Under the same distribution it can come out
    # slightly negative (that is normal).
    n, m = len(X), len(Y)
    Kxx, Kyy, Kxy = k(X, X, sigma), k(Y, Y, sigma), k(X, Y, sigma)
    sxx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) if n > 1 else 0.0
    syy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) if m > 1 else 0.0
    return float(sxx + syy - 2 * Kxy.mean())


def _frechet(X, Y):
    # Isomorphic to FID. The trace of sqrtm(cx@cy) is computed as the sum of the
    # sqrt of eigvals(cx@cy) (pure numpy, no scipy).
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    mx, my = X.mean(0), Y.mean(0)
    cx, cy = np.cov(X, rowvar=False), np.cov(Y, rowvar=False)
    diff = mx - my
    eigs = np.linalg.eigvals(cx @ cy)
    tr_sqrt = float(np.sum(np.sqrt(np.maximum(eigs.real, 0.0))))
    return float(diff @ diff + np.trace(cx) + np.trace(cy) - 2 * tr_sqrt)


async def main_async(args):
    import soundfile as sf
    embed = _clap_embed_fn()
    mix_dir = ROOT / "outputs/runs" / args.run / "mixes"
    by_method = defaultdict(list)
    excl = [s for s in args.exclude_substr.split(",") if s]
    for f in glob.glob(str(mix_dir / "*.wav")):
        if any(s in Path(f).name for s in excl):
            continue
        method = Path(f).name.split("__")[0]
        x, sr = sf.read(f); x = x.T if x.ndim == 2 else np.stack([x, x])
        by_method[method].append(embed(np.asarray(x, np.float32), sr))
    print(f"[kad] methods: {[(m, len(v)) for m, v in by_method.items()]}")

    ref_emb = []
    for r in enumerate_split("train", tiers=None, limit=args.ref_n):
        try:
            stems, sr = load_one(dataset="musdb18", song_id=r["song_id"], duration_sec=args.duration, split="train")
            mix = stems.get("mixture")
            if mix is None:
                mix = np.sum([v for k, v in stems.items() if k != "mixture"], axis=0)
            ref_emb.append(embed(np.asarray(mix, np.float32), sr))
        except Exception as ex:                                # noqa: BLE001
            print(f"  ref fail: {ex}")
    ref = np.array(ref_emb)
    print(f"[kad] reference n={len(ref)}")

    print("\n=== KAD / FAD (vs MUSDB pro dist; lower = better) ===")
    print(f"  {'method':8s} {'KAD':>12s} {'FAD':>12s} {'n':>4s}")
    out = {}
    for m, embs in sorted(by_method.items()):
        X = np.array(embs)
        kad = _mmd_gaussian(X, ref) if len(X) >= 2 else float("nan")
        try:
            fad = _frechet(X, ref) if len(X) >= 2 else float("nan")
        except Exception:                                      # noqa: BLE001
            fad = float("nan")
        out[m] = {"KAD": kad, "FAD": fad, "n": len(X)}
        print(f"  {m:8s} {kad:>12.5f} {fad:>12.3f} {len(X):>4d}")
    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    (ROOT / "outputs/runs" / args.run / f"kad_results{suffix}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[kad] note: small sample (n={len(next(iter(by_method.values()),[]))}) -> indicative only. A large sample is needed to trust it.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="all_metrics_v1")
    p.add_argument("--ref-n", type=int, default=50)
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--exclude-substr", default="",
                   help="comma-separated: exclude wavs whose file name contains any of these (e.g. Lushlife)")
    p.add_argument("--out-suffix", default="",
                   help="write the output to kad_results_<suffix>.json (default is kad_results.json)")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
