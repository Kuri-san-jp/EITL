"""Pairwise A/B listening test (Gradio).

For each trial, two anonymised mixes are presented; the listener picks
"A", "B", or "Tie", optionally with a free-text reason. Records go to JSONL.

stimuli.json schema:
  [{"song_id":"...","pair":["a.wav","b.wav"],"systems":["ours","bo"]}, ...]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List


def _build_app(stimuli: List[Dict[str, Any]], out_path: Path):
    import gradio as gr

    state = {"listener_id": "", "trial_idx": 0, "stimuli": stimuli, "current": None}

    def begin(listener_id: str):
        if not listener_id.strip():
            return ("Enter your listener ID.", None, None, gr.update(visible=False))
        state["listener_id"] = listener_id.strip()
        state["trial_idx"] = 0
        return _render_trial()

    def _render_trial():
        i = state["trial_idx"]
        if i >= len(state["stimuli"]):
            return ("All trials complete — thank you!", None, None, gr.update(visible=False))
        s = state["stimuli"][i]
        # randomize A/B order
        order = list(range(2))
        random.shuffle(order)
        a_path = s["pair"][order[0]]
        b_path = s["pair"][order[1]]
        state["current"] = {"i": i, "order": order, "stim": s}
        text = f"Trial {i+1}/{len(state['stimuli'])} — Song: {s.get('song_id','?')}\nWhich do you prefer?"
        return text, a_path, b_path, gr.update(visible=True)

    def submit(pref: str, reason: str):
        c = state["current"]
        if c is None:
            return ("Click Begin first.", None, None, gr.update(visible=False))
        s = c["stim"]; order = c["order"]
        # Decode preference back to system labels
        if pref == "A":
            preferred = s["systems"][order[0]]
        elif pref == "B":
            preferred = s["systems"][order[1]]
        else:
            preferred = "tie"
        rec = {
            "listener_id": state["listener_id"],
            "timestamp": time.time(),
            "song_id": s.get("song_id"),
            "trial_idx": c["i"],
            "shown_order": [s["systems"][k] for k in order],
            "preferred": preferred,
            "reason": reason,
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        state["trial_idx"] += 1
        return _render_trial()

    with gr.Blocks(title="Pairwise listening test") as app:
        gr.Markdown("# Pairwise A/B test")
        listener = gr.Textbox(label="Listener ID")
        start_btn = gr.Button("Begin")
        status = gr.Markdown("Enter a listener ID and click Begin.")
        audio_a = gr.Audio(label="A")
        audio_b = gr.Audio(label="B")
        trial_box = gr.Group(visible=False)
        with trial_box:
            pref = gr.Radio(choices=["A", "B", "Tie"], label="Preference")
            reason = gr.Textbox(label="(optional) reason")
            submit_btn = gr.Button("Submit")
        start_btn.click(begin, inputs=[listener],
                        outputs=[status, audio_a, audio_b, trial_box])
        submit_btn.click(submit, inputs=[pref, reason],
                         outputs=[status, audio_a, audio_b, trial_box])
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", required=True)
    ap.add_argument("--out", default="outputs/human_eval/pairwise.jsonl")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    stimuli = json.loads(Path(args.stimuli).read_text("utf-8"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    app = _build_app(stimuli, out)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
