"""Hugging Face Space: System 1 of the Smart Request Router (SetFit + escalation policy), CPU only.

It runs the same `decide()` as the services (routercore) on the published SetFit model. Messages that
System 1 would not answer show up as "escalates to System 2" with the reason; the agent is not part of
this demo.
MODEL_DIR (env) overrides the download, used to test locally.
"""
import json
import os
from pathlib import Path

import gradio as gr
import yaml

from routercore.domain.policy import Policy, decide

MODEL_REPO = "WhiteUniversal/smart-request-router-setfit"
URGENCY = ["0", "1", "2", "3"]
EXAMPLES = ["I lost my card and someone is using it", "Why was I charged a fee on my account?",
            "Quiero cambiar mi dirección y también cancelar mi tarjeta", "hello",
            "How do I top up with Apple Pay?"]


def load(model_dir: str | None = None):
    from huggingface_hub import snapshot_download
    from setfit import SetFitModel

    path = Path(model_dir or os.environ.get("MODEL_DIR") or snapshot_download(MODEL_REPO))
    labels = json.loads((path / "labels.json").read_text(encoding="utf-8"))["intent"]
    raw = yaml.safe_load((path / "policy.yaml").read_text(encoding="utf-8"))
    policy = Policy(tau=raw["tau"], margin=raw["margin"], tau_urgent=raw["tau_urgent"],
                    max_entropy=raw["max_entropy"])
    return (SetFitModel.from_pretrained(str(path / "intent")), labels,
            SetFitModel.from_pretrained(str(path / "urgency")), policy)


def make_router(intent_model, labels, urgency_model, policy):
    def route(text: str):
        if not text.strip():
            return "", {}, {}
        route_p = dict(zip(labels, intent_model.predict_proba([text])[0].tolist(), strict=True))
        urgency_p = dict(zip(URGENCY, urgency_model.predict_proba([text])[0].tolist(), strict=True))
        target, reason = decide(route_p, urgency_p, True, policy)
        verdict = (f"System 1 answers: **{target}**" if target != "system2"
                   else f"Escalates to System 2 (reason: `{reason}`)")
        return verdict, route_p, {f"urgency {k}": v for k, v in urgency_p.items()}
    return route


def build_app() -> gr.Blocks:
    route = make_router(*load())
    with gr.Blocks(title="Smart Request Router - System 1") as demo:
        gr.Markdown("# Smart Request Router - System 1\nA SetFit model with an escalation policy for bank "
                    "support messages (English / Spanish). It answers when confident and escalates otherwise. "
                    "Code: https://github.com/Leito2/smart-request-router")
        text = gr.Textbox(label="Customer message", lines=2)
        verdict = gr.Markdown()
        with gr.Row():
            routes = gr.Label(label="Route probabilities", num_top_classes=5)
            urgency = gr.Label(label="Urgency probabilities")
        text.submit(route, text, [verdict, routes, urgency])
        gr.Button("Route").click(route, text, [verdict, routes, urgency])
        gr.Examples(EXAMPLES, text)
    return demo


if __name__ == "__main__":
    build_app().launch()
