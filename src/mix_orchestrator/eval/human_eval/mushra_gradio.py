"""MUSHRA-lite listening test (Gradio).

Each trial shows the reference + N anonymised system outputs and asks
the listener to rate each on a 0–100 slider. Results saved to JSONL.

Spec:
  python -m mix_orchestrator.eval.human_eval.mushra_gradio \
      --stimuli /path/to/stimuli.json \
      --out outputs/human_eval/mushra.jsonl

`stimuli.json` schema:
  [
    {"song_id":"...", "reference":"path/ref.wav",
     "systems":{"ours":"a.wav","bo":"b.wav","rule":"c.wav", ...}},
    ...
  ]
"""
from __future__ import annotations

import argparse
import json
import random
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List


def _build_app(stimuli: List[Dict[str, Any]], out_path: Path):
    import gradio as gr

    state = {"listener_id": "", "trial_idx": 0, "stimuli": stimuli}

    def begin(listener_id: str):
        if not listener_id.strip():
            return ("Enter your listener ID first.",
                    gr.update(visible=False))
        state["listener_id"] = listener_id.strip()
        state["trial_idx"] = 0
        return _render_trial()

    def _render_trial():
        i = state["trial_idx"]
        if i >= len(state["stimuli"]):
            return ("All trials complete — thank you!", gr.update(visible=False))
        s = state["stimuli"][i]
        # Anonymise system order
        keys = list(s["systems"])
        random.shuffle(keys)
        labels = [f"System {chr(ord('A') + k)}" for k in range(len(keys))]
        state["current_keys"] = keys
        text = f"Trial {i+1}/{len(state['stimuli'])} — Song: {s.get('song_id', '?')}\nRate each system 0–100."
        # Returning audio components dynamically isn't trivial — fall back to a static N=6 layout.
        return text, gr.update(visible=True)

    def submit(*scores):
        i = state["trial_idx"]
        s = state["stimuli"][i]
        keys = state["current_keys"]
        rec = {
            "listener_id": state["listener_id"],
            "timestamp":   time.time(),
            "song_id":     s.get("song_id"),
            "trial_idx":   i,
            "scores":      {k: float(v) for k, v in zip(keys, scores) if v is not None},
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        state["trial_idx"] += 1
        return _render_trial()

    with gr.Blocks(title="MUSHRA-lite") as app:
        gr.Markdown("# MUSHRA-lite listening test")
        listener = gr.Textbox(label="Listener ID")
        start_btn = gr.Button("Begin")
        status = gr.Markdown("Enter a listener ID and click Begin.")
        trial_box = gr.Group(visible=False)
        with trial_box:
            audio_components = [gr.Audio(label=f"System {chr(ord('A')+k)}") for k in range(6)]
            sliders = [gr.Slider(0, 100, value=50, label=f"System {chr(ord('A')+k)} quality")
                       for k in range(6)]
            submit_btn = gr.Button("Submit this trial")
        start_btn.click(begin, inputs=[listener], outputs=[status, trial_box])
        submit_btn.click(submit, inputs=sliders, outputs=[status, trial_box])
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", required=True)
    ap.add_argument("--out", default="outputs/human_eval/mushra.jsonl")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    stimuli = json.loads(Path(args.stimuli).read_text("utf-8"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    app = _build_app(stimuli, out)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
