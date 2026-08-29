# EITL: Ear-in-the-Loop Music Mixing

Code for the ICASSP 2027 submission *"EITL: Ear-in-the-Loop Music Mixing
Measures and Mitigates Reward Hacking of Learned Quality Metrics."*

Anonymised for double-anonymous review. Author names, affiliations, absolute
paths, and the listening-study operational material (completion codes,
screening answer keys, platform tokens) are removed.

## What this is

An eval-gated greedy search over DSP actions for automatic mixing, scored by
two reference-free learned metrics: the production-quality axis of
Audiobox-Aesthetics (`PQ`) and the Mixing axis of SongBench (`SB`). The point
of the paper is what happens when such a metric becomes the optimisation
target - and that gating on the worst case of two independently trained
metrics removes the systematic sacrifice of whichever axis a single-metric
reward ignores.

## Layout

```
src/mix_orchestrator/
  dsp/          mix state, DSP actions, loudness normalisation
  eval/         reward forms (min / mean / single-metric), per-song calibration
  strategies/   knowledge-based gain initialisation
  ears/         scorer wrappers
experiments/
  agreement_loop_all.py    the search loop; corruption, calibration, gating
  agreement_loop_qwen.py   LLM proposer
  size_sweep_run.py        budget-sweep driver (k up to 10^4)
  kad_uniform_det.py       held-out KAD, deterministic and reference-shared
  subjective/pilot_analysis.py   mixed-model analysis of the listening data
tools/
  scoring_server.py        metric server (keeps models resident across songs)
  zdelta_rewardform.py     pre/post change per reward form, in calibrated units
  contrast_mixed.py        condition contrasts under the listening-test model
prompts/                   frozen LLM proposer prompts
```

## Reward

Each candidate render is scored by both models. Scores are z-calibrated per
song against a pool containing the initial mix and twelve perturbed variants,
then aggregated. The four reward forms compared in the paper are the minimum
(ours), a weighted mean, and each metric alone. An edit is accepted only on
strict improvement of the aggregate.

## Running

Audio is not included; MUSDB18-HQ is obtained separately from Zenodo. Metric
checkpoints are pulled from their upstream sources on first use. The search
expects a scoring server:

```bash
python3 tools/scoring_server.py --port 9600 --strict
python3 experiments/size_sweep_run.py \
    --splits train,dev,test --limit 100 --random-seeds 2 \
    --random-gain-db-range -43 -18 --normalize-corrupted-lufs -14 \
    --n-search 28 --reward-form min --proposer random \
    --scorer remote --scorer-addr 127.0.0.1:9600 --run-name demo
```

`--reward-form` takes `min`, `weighted`, `pq_only`, or `sb_only`.
`--n-search` is the budget `k`; the sweeps in the paper run to `10^4` with
`--search-excerpt-sec 12.0`.

## Reproducibility notes

Corruption gains derive from `(song_index, seed_index)` through
`random_seed_for`, so a run is reproducible from its recorded song order.
The KAD diagnostic computes its reference embeddings once and shares them
across arms; identical inputs give bit-identical KAD across runs, which the
script checks by measuring the corrupted baseline in several runs at once.

The listening-test analysis script is included, but the response data are not:
they are participant records from a crowdsourcing platform and are outside
what this release covers.

Source comments are in Japanese in places; this is the working codebase with
identifying material removed, not a rewrite.
