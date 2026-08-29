"""Gradio UI for the E4 diagnosis Likert task.

Shows one diagnosis at a time. The rater picks a 1-5 Likert score
("not a valid mixing problem to address" .. "exactly the right thing
to address"). System labels are HIDDEN — the rater only sees text.

Spec:
    python -m mix_orchestrator.eval.human_eval.diagnosis_likert_gradio \
        --pool outputs/runs/e4_likert/diagnoses.jsonl \
        --out  outputs/runs/e4_likert/ratings.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List


LIKERT_LABELS = {
    1: "1 — not a valid mixing problem",
    2: "2 — possibly relevant but weak",
    3: "3 — neutral",
    4: "4 — plausible mixing concern",
    5: "5 — exactly the right thing to address",
}


def _build_app(pool: List[Dict[str, Any]], out_path: Path):
    import gradio as gr

    state = {"rater_id": "", "idx": 0, "pool": pool}

    def begin(rater_id: str):
        if not rater_id.strip():
            return ("Enter your rater ID.", "", gr.update(visible=False))
        state["rater_id"] = rater_id.strip()
        state["idx"] = 0
        random.shuffle(state["pool"])
        return _render()

    def _render():
        i = state["idx"]
        if i >= len(state["pool"]):
            return ("All done — thank you!", "", gr.update(visible=False))
        item = state["pool"][i]
        msg = f"Item {i+1}/{len(state['pool'])}"
        return (msg, item["diagnosis"], gr.update(visible=True))

    def submit(score, comment):
        i = state["idx"]
        if i >= len(state["pool"]):
            return ("Already finished.", "", gr.update(visible=False))
        item = state["pool"][i]
        rec = {
            "rater_id": state["rater_id"],
            "timestamp": time.time(),
            "item_id": item["id"],
            "system": item["system"],         # ground truth, hidden from rater
            "source": item["source"],
            "score": int(score) if score else None,
            "comment": comment,
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        state["idx"] += 1
        return _render()

    with gr.Blocks(title="Diagnosis Likert task") as app:
        gr.Markdown("# Mix diagnosis — Likert task")
        gr.Markdown(
            "You will be shown a series of short diagnoses written by various "
            "mixing systems (some by humans, some by software). Rate each "
            "diagnosis on a 1–5 scale based on whether it identifies a *real* "
            "mixing problem worth addressing."
        )
        rater = gr.Textbox(label="Rater ID")
        start_btn = gr.Button("Begin")
        status = gr.Markdown("Enter rater ID and click Begin.")
        diagnosis = gr.Textbox(label="Diagnosis", lines=4, interactive=False)
        trial = gr.Group(visible=False)
        with trial:
            score = gr.Radio(choices=list(LIKERT_LABELS.values()), label="Score")
            comment = gr.Textbox(label="(optional) comment", lines=2)
            submit_btn = gr.Button("Submit")

        def _decode_score(label_str):
            for k, v in LIKERT_LABELS.items():
                if v == label_str:
                    return k
            return None

        start_btn.click(begin, inputs=[rater], outputs=[status, diagnosis, trial])
        submit_btn.click(
            lambda s, c: submit(_decode_score(s), c),
            inputs=[score, comment],
            outputs=[status, diagnosis, trial],
        )
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="diagnoses.jsonl from run_e4_diagnosis_extract.py")
    ap.add_argument("--out", default="outputs/runs/e4_likert/ratings.jsonl")
    ap.add_argument("--port", type=int, default=7862)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    pool = [json.loads(l) for l in Path(args.pool).read_text("utf-8").splitlines() if l.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    app = _build_app(pool, out)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
