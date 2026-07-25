from __future__ import annotations

import html
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
from huggingface_hub import InferenceClient


APP_TITLE = "Feeling Compass"
HOSTED_MODEL_ID = os.getenv(
    "HF_MODEL_ID",
    "j-hartmann/emotion-english-distilroberta-base",
)
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_MODEL_DIR = PROJECT_ROOT / "checkpoints" / "distilbert_emotion_model"
LOCAL_TOKENIZER_DIR = PROJECT_ROOT / "checkpoints" / "distilbert_emotion_tokenizer"
MAX_TEXT_LENGTH = 3_500

EMOTIONS = {
    "anger": {
        "name": "Anger",
        "color": "#FF6268",
        "meaning": "The words may carry frustration, unfairness, or a crossed boundary.",
    },
    "disgust": {
        "name": "Disgust",
        "color": "#75D48B",
        "meaning": "The wording may signal rejection, distaste, or a strong wish to pull away.",
    },
    "fear": {
        "name": "Fear",
        "color": "#8F9CFF",
        "meaning": "The message may hold worry, uncertainty, or a need for reassurance.",
    },
    "joy": {
        "name": "Joy",
        "color": "#F2C85B",
        "meaning": "The tone feels pleased, hopeful, grateful, or warmly excited.",
    },
    "neutral": {
        "name": "Even",
        "color": "#A9B2C3",
        "meaning": "The words feel steady and matter-of-fact, without one strong emotion leading.",
    },
    "sadness": {
        "name": "Sadness",
        "color": "#55BDEB",
        "meaning": "The message may carry disappointment, loss, regret, or emotional heaviness.",
    },
    "surprise": {
        "name": "Surprise",
        "color": "#D58AF1",
        "meaning": "The wording reacts to something unexpected, sudden, or hard to anticipate.",
    },
}

EMOTION_ORDER = (
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
)

HOSTED_LABEL_ALIASES = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "joy",
    "neutral": "neutral",
    "sadness": "sadness",
    "surprise": "surprise",
    "label_0": "anger",
    "label_1": "disgust",
    "label_2": "fear",
    "label_3": "joy",
    "label_4": "neutral",
    "label_5": "sadness",
    "label_6": "surprise",
}

LOCAL_LABEL_ALIASES = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "joy",
    "sadness": "sadness",
    "surprise": "surprise",
    "label_0": "anger",
    "label_1": "disgust",
    "label_2": "fear",
    "label_3": "joy",
    "label_4": "sadness",
    "label_5": "surprise",
}

NEXT_STEPS = {
    "A message": {
        "anger": "Acknowledge the concern first, then ask what outcome would feel fair.",
        "disgust": "Give the person room to explain what felt wrong before offering a fix.",
        "fear": "Lower the pressure and offer one clear, manageable next step.",
        "joy": "Reflect the good feeling back with a warm and specific reply.",
        "neutral": "Reply directly to the main point and keep the same steady tone.",
        "sadness": "Recognize the feeling before moving toward advice or solutions.",
        "surprise": "Ask for a little more context before deciding whether the surprise is welcome.",
    },
    "A personal note": {
        "anger": "Name what crossed a boundary and what you need to feel settled.",
        "disgust": "Notice what you want distance from and why it matters to you.",
        "fear": "Separate what you know from what you are imagining, then choose one next step.",
        "joy": "Pause on what feels good and what you want to remember about it.",
        "neutral": "The note sounds settled. More detail may reveal a quieter feeling underneath.",
        "sadness": "Let the feeling be present without forcing an immediate solution.",
        "surprise": "Write down what changed and what you still want to understand.",
    },
    "Feedback": {
        "anger": "Look for the unmet expectation beneath the frustration before responding.",
        "disgust": "Identify the specific experience that caused such a strong rejection.",
        "fear": "Clarify the risk the writer is worried about and address it directly.",
        "joy": "Preserve the detail that delighted the writer and build on it.",
        "neutral": "Focus on the concrete request or observation in the feedback.",
        "sadness": "Acknowledge the disappointment and explain the next helpful action.",
        "surprise": "Confirm what differed from expectations before drawing a conclusion.",
    },
}

SAMPLES = {
    "tense": "I explained why this mattered, but it was ignored again and I am tired of having to repeat myself.",
    "bright": "That was such a thoughtful surprise. I have been smiling about it all afternoon.",
    "uncertain": "I keep thinking about tomorrow and worrying that I made the wrong choice.",
}


@lru_cache(maxsize=1)
def get_hosted_client() -> InferenceClient:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    return InferenceClient(
        model=HOSTED_MODEL_ID,
        provider="hf-inference",
        token=token,
        timeout=45,
    )


@lru_cache(maxsize=1)
def get_local_model() -> tuple[Any, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(LOCAL_TOKENIZER_DIR),
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(LOCAL_MODEL_DIR),
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def empty_scores() -> dict[str, float]:
    return {emotion: 0.0 for emotion in EMOTION_ORDER}


def read_item(item: Any) -> tuple[str, float]:
    if isinstance(item, dict):
        label = item.get("label", "")
        score = item.get("score", 0.0)
    else:
        label = getattr(item, "label", "")
        score = getattr(item, "score", 0.0)
    return str(label).strip().lower(), float(score)


def normalize_results(
    results: list[Any],
    aliases: dict[str, str],
) -> dict[str, float]:
    scores = empty_scores()

    for item in results:
        raw_label, score = read_item(item)
        emotion = aliases.get(raw_label)
        if emotion:
            scores[emotion] += max(0.0, score)

    total = sum(scores.values())
    if total <= 0:
        raise ValueError("No readable emotion scores were returned.")

    return {emotion: score / total for emotion, score in scores.items()}


def classify_with_local_model(text: str) -> dict[str, float]:
    import torch

    tokenizer, model = get_local_model()
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.inference_mode():
        logits = model(**encoded).logits[0]
        probabilities = torch.softmax(logits, dim=-1).tolist()

    results = [
        {
            "label": model.config.id2label.get(index, f"LABEL_{index}"),
            "score": score,
        }
        for index, score in enumerate(probabilities)
    ]
    return normalize_results(results, LOCAL_LABEL_ALIASES)


def classify_with_hosted_model(text: str) -> dict[str, float]:
    results = get_hosted_client().text_classification(text, top_k=None)
    return normalize_results(results, HOSTED_LABEL_ALIASES)


def classify_text(text: str) -> dict[str, float]:
    if LOCAL_MODEL_DIR.is_dir() and LOCAL_TOKENIZER_DIR.is_dir():
        try:
            return classify_with_local_model(text)
        except Exception:
            pass

    return classify_with_hosted_model(text)


def build_ring(scores: dict[str, float]) -> str:
    if sum(scores.values()) <= 0:
        return "conic-gradient(#2A303B 0% 100%)"

    segments: list[str] = []
    start = 0.0
    for index, emotion in enumerate(EMOTION_ORDER):
        end = 100.0 if index == len(EMOTION_ORDER) - 1 else start + scores[emotion] * 100
        color = EMOTIONS[emotion]["color"]
        segments.append(f"{color} {start:.2f}% {end:.2f}%")
        start = end
    return f"conic-gradient({', '.join(segments)})"


def render_waiting_reading() -> str:
    return """
    <section class="reading-state is-empty" aria-live="polite">
        <div class="compass-visual" style="--emotion-ring: conic-gradient(#2A303B 0% 100%);">
            <div class="compass-ring">
                <div class="compass-core">
                    <span>Overall tone</span>
                    <strong>Waiting</strong>
                </div>
            </div>
        </div>
        <div class="reading-copy">
            <span class="section-label">Your reading</span>
            <h2>The emotional direction will appear here.</h2>
            <p>Choose a piece of writing above to reveal its strongest feeling and the quieter tones around it.</p>
        </div>
    </section>
    """


def render_notice(title: str, message: str) -> str:
    return f"""
    <section class="reading-state is-empty" aria-live="polite">
        <div class="compass-visual" style="--emotion-ring: conic-gradient(#2A303B 0% 100%);">
            <div class="compass-ring">
                <div class="compass-core">
                    <span>Not read yet</span>
                    <strong>Pause</strong>
                </div>
            </div>
        </div>
        <div class="reading-copy">
            <span class="section-label">A quick check</span>
            <h2>{html.escape(title)}</h2>
            <p>{html.escape(message)}</p>
        </div>
    </section>
    """


def render_reading(scores: dict[str, float], context: str) -> str:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    primary, primary_score = ranked[0]
    secondary, secondary_score = ranked[1]

    primary_name = EMOTIONS[primary]["name"]
    secondary_name = EMOTIONS[secondary]["name"]
    primary_color = EMOTIONS[primary]["color"]
    meaning = EMOTIONS[primary]["meaning"]
    next_step = NEXT_STEPS[context][primary]

    if secondary_score >= 0.18:
        headline = f"{primary_name} leads, with {secondary_name.lower()} nearby."
    elif primary_score < 0.42:
        headline = f"The tone feels mixed, leaning toward {primary_name.lower()}."
    else:
        headline = f"{primary_name} is the clearest feeling."

    return f"""
    <section class="reading-state" aria-live="polite">
        <div class="compass-visual" style="
            --emotion-ring: {build_ring(scores)};
            --primary-color: {primary_color};
        ">
            <div class="compass-ring">
                <div class="compass-core">
                    <span>Overall tone</span>
                    <strong>{html.escape(primary_name)}</strong>
                </div>
            </div>
        </div>
        <div class="reading-copy">
            <span class="section-label">Your reading</span>
            <h2>{html.escape(headline)}</h2>
            <p>{html.escape(meaning)}</p>
            <div class="next-step">
                <span>A thoughtful next move</span>
                <p>{html.escape(next_step)}</p>
            </div>
        </div>
    </section>
    """


def render_profile(scores: dict[str, float] | None = None) -> str:
    values = scores or empty_scores()
    has_values = sum(values.values()) > 0

    rows: list[str] = []
    for emotion in EMOTION_ORDER:
        details = EMOTIONS[emotion]
        percentage = round(values[emotion] * 100)
        width = max(percentage, 2) if has_values else 0
        value_text = f"{percentage}%" if has_values else "--"
        rows.append(
            f"""
            <div class="profile-row">
                <div class="profile-meta">
                    <span class="emotion-dot" style="--dot-color: {details['color']};"></span>
                    <span>{html.escape(details['name'])}</span>
                    <strong>{value_text}</strong>
                </div>
                <div class="profile-track" aria-hidden="true">
                    <span style="--bar-color: {details['color']}; width: {width}%;"></span>
                </div>
            </div>
            """
        )

    note = (
        "The circle and bars show how the feelings blend together."
        if has_values
        else "The full blend will appear after the first reading."
    )

    return f"""
    <section class="profile-content" aria-live="polite">
        <div class="profile-heading">
            <div>
                <span class="section-label">Feeling blend</span>
                <h2>Signals in the words</h2>
            </div>
            <p>{note}</p>
        </div>
        <div class="profile-list">
            {''.join(rows)}
        </div>
    </section>
    """


def analyze_text(text: str) -> tuple[str, str]:
    cleaned = " ".join((text or "").split())

    if not cleaned:
        return (
            render_notice(
                "Add some words first.",
                "A complete sentence or two usually gives the clearest reading.",
            ),
            render_profile(),
        )

    if len(cleaned.split()) < 3:
        return (
            render_notice(
                "A little more would help.",
                "Add a few more words so the emotional tone has enough context.",
            ),
            render_profile(),
        )

    cleaned = cleaned[:MAX_TEXT_LENGTH]

    try:
        scores = classify_text(cleaned)
    except Exception:
        return (
            render_notice(
                "The reader is unavailable right now.",
                "Your words are still here. Wait a moment, then try the reading again.",
            ),
            render_profile(),
        )

    return render_reading(scores, "A message"), render_profile(scores)


def clear_app() -> tuple[str, str, str]:
    return "", render_waiting_reading(), render_profile()


THEME = gr.themes.Soft(
    primary_hue="amber",
    secondary_hue="slate",
    neutral_hue="slate",
).set(
    body_background_fill="#090B10",
    body_background_fill_dark="#090B10",
    body_text_color="#F4F6FA",
    body_text_color_dark="#F4F6FA",
    body_text_color_subdued="#98A2B3",
    body_text_color_subdued_dark="#98A2B3",
    background_fill_primary="#11151D",
    background_fill_primary_dark="#11151D",
    background_fill_secondary="#171C26",
    background_fill_secondary_dark="#171C26",
    border_color_primary="#2A303B",
    border_color_primary_dark="#2A303B",
    block_background_fill="#11151D",
    block_background_fill_dark="#11151D",
    block_border_color="#2A303B",
    block_border_color_dark="#2A303B",
    block_label_text_color="#F4F6FA",
    block_label_text_color_dark="#F4F6FA",
    input_background_fill="#0D1016",
    input_background_fill_dark="#0D1016",
    input_border_color="#343B48",
    input_border_color_dark="#343B48",
    input_placeholder_color="#6F7A8C",
    input_placeholder_color_dark="#6F7A8C",
    button_primary_background_fill="#F2C85B",
    button_primary_background_fill_dark="#F2C85B",
    button_primary_background_fill_hover="#FFDA75",
    button_primary_background_fill_hover_dark="#FFDA75",
    button_primary_text_color="#111318",
    button_primary_text_color_dark="#111318",
    button_secondary_background_fill="#171C26",
    button_secondary_background_fill_dark="#171C26",
    button_secondary_background_fill_hover="#202632",
    button_secondary_background_fill_hover_dark="#202632",
    button_secondary_text_color="#E9EDF5",
    button_secondary_text_color_dark="#E9EDF5",
    button_secondary_border_color="#343B48",
    button_secondary_border_color_dark="#343B48",
    shadow_drop="none",
    shadow_drop_lg="none",
    block_shadow="none",
)


CSS = """
:root {
    color-scheme: dark;
    --page: #090B10;
    --surface: #11151D;
    --surface-raised: #171C26;
    --line: #2A303B;
    --line-strong: #3A4250;
    --text: #F4F6FA;
    --muted: #98A2B3;
    --faint: #6F7A8C;
    --action: #F2C85B;
}

html,
body,
gradio-app,
.gradio-container {
    width: 100%;
    height: 100%;
    min-height: 0 !important;
    background: var(--page) !important;
    overflow: hidden !important;
}

body {
    margin: 0;
}

.gradio-container {
    height: 100dvh !important;
    max-width: none !important;
    padding: 0 !important;
    color: var(--text) !important;
    font-family: "Segoe UI Variable Text", "Segoe UI", Arial, sans-serif !important;
    overflow: hidden !important;
}

.gradio-container * {
    box-sizing: border-box;
    letter-spacing: 0 !important;
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container p {
    margin-top: 0;
}

.gradio-container footer,
.gradio-container .built-with {
    display: none !important;
}

.page-shell {
    width: 100%;
    height: 100dvh;
    min-height: 0 !important;
    padding: 0 30px 16px;
    gap: 12px !important;
    overflow: hidden;
}

#app-header {
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
}

.app-header {
    height: 68px;
    min-height: 68px;
    flex: 0 0 68px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    border-bottom: 1px solid var(--line);
}

.brand {
    display: flex;
    align-items: center;
    gap: 13px;
}

.brand-mark {
    position: relative;
    width: 36px;
    height: 36px;
    flex: 0 0 36px;
    border: 1px solid var(--line-strong);
    border-radius: 50%;
}

.brand-mark::before,
.brand-mark::after {
    content: "";
    position: absolute;
    inset: 50% auto auto 50%;
    background: var(--action);
    transform: translate(-50%, -50%);
}

.brand-mark::before {
    width: 16px;
    height: 2px;
}

.brand-mark::after {
    width: 2px;
    height: 16px;
}

.brand-mark span {
    position: absolute;
    top: 7px;
    right: 7px;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #D58AF1;
    box-shadow: 0 0 0 3px var(--page);
}

.brand-copy h1 {
    margin: 0;
    color: var(--text);
    font-family: "Segoe UI Variable Display", "Segoe UI", Arial, sans-serif;
    font-size: 22px;
    line-height: 1.05;
    font-weight: 720;
}

.brand-copy p {
    margin: 4px 0 0;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.3;
}

.header-note {
    display: flex;
    align-items: center;
    gap: 9px;
    color: var(--muted);
    font-size: 13px;
    white-space: nowrap;
}

.header-note span {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #75D48B;
    box-shadow: 0 0 0 4px rgba(117, 212, 139, 0.10);
}

.surface-panel {
    border: 1px solid var(--line) !important;
    border-radius: 8px !important;
    background: var(--surface) !important;
    overflow: hidden;
}

.composer-panel {
    flex: 0 0 clamp(268px, 34vh, 320px);
    min-height: 0 !important;
    padding: 18px !important;
    gap: 12px !important;
}

#composer-heading {
    flex: 0 0 auto;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
}

.composer-heading {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 24px;
}

.section-label {
    display: inline-block;
    margin-bottom: 8px;
    color: var(--muted);
    font-size: 11px;
    line-height: 1;
    font-weight: 700;
    text-transform: uppercase;
}

.composer-heading h2,
.profile-heading h2 {
    margin: 0;
    color: var(--text);
    font-family: "Segoe UI Variable Display", "Segoe UI", Arial, sans-serif;
    font-size: 23px;
    line-height: 1.15;
    font-weight: 680;
}

.composer-heading p {
    max-width: 430px;
    margin: 0;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.5;
    text-align: right;
}

.composer-grid {
    flex: 1 1 auto;
    min-height: 0 !important;
    align-items: stretch !important;
    gap: 14px !important;
    overflow: hidden;
}

.composer-side,
.composer-actions {
    height: 100%;
    min-height: 0 !important;
    justify-content: space-between;
    gap: 10px !important;
}

.mini-label {
    margin: 0 0 8px !important;
    color: var(--muted) !important;
    font-size: 12px !important;
    font-weight: 650 !important;
}

.sample-stack {
    gap: 5px !important;
}

.sample-button,
#clear-button {
    min-height: 34px !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    font-weight: 620 !important;
}

.sample-button {
    justify-content: flex-start !important;
    text-align: left !important;
}

#message-input {
    height: 100%;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 1px solid var(--line-strong) !important;
    border-radius: 8px !important;
    background: #0D1016 !important;
    box-shadow: none !important;
    overflow: hidden !important;
}

#message-input::before,
#message-input::after {
    display: none !important;
    content: none !important;
}

#message-input label {
    height: 100%;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

#message-input .wrap {
    height: 100%;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

#message-input .input-container {
    height: 100%;
    min-height: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

#message-input textarea {
    height: 100% !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 16px !important;
    border: 0 !important;
    border-radius: 7px !important;
    outline: 0 !important;
    background: #0D1016 !important;
    box-shadow: none !important;
    color: var(--text) !important;
    font-family: "Segoe UI Variable Text", "Segoe UI", Arial, sans-serif !important;
    font-size: 17px !important;
    line-height: 1.65 !important;
    resize: none !important;
}

#message-input textarea:focus,
#message-input textarea:focus-visible {
    border: 0 !important;
    outline: 0 !important;
    box-shadow: none !important;
}

#message-input textarea:placeholder-shown {
    caret-color: transparent !important;
}

#message-input textarea:not(:placeholder-shown) {
    caret-color: var(--text) !important;
}

#message-input:focus-within {
    border-color: #596272 !important;
    box-shadow: none !important;
}

#read-button {
    min-height: 76px !important;
    border: 0 !important;
    border-radius: 8px !important;
    font-size: 16px !important;
    font-weight: 760 !important;
}

.action-note {
    margin: 0 !important;
    color: var(--faint) !important;
    font-size: 12px !important;
    line-height: 1.5 !important;
}

.results-row {
    flex: 1 1 auto;
    min-height: 0 !important;
    align-items: stretch !important;
    gap: 12px !important;
    overflow: hidden;
}

.reading-panel,
.profile-panel {
    height: 100%;
    min-height: 0 !important;
}

.reading-panel {
    padding: 20px !important;
}

.profile-panel {
    padding: 20px !important;
}

#reading-output,
#profile-output {
    height: 100%;
    min-height: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
}

.reading-state {
    height: 100%;
    min-height: 0;
    display: grid;
    grid-template-columns: minmax(190px, 0.8fr) minmax(280px, 1.3fr);
    align-items: center;
    gap: clamp(24px, 4vw, 58px);
}

.compass-visual {
    display: grid;
    place-items: center;
}

.compass-ring {
    width: min(21vh, 196px, 100%);
    aspect-ratio: 1;
    padding: 15px;
    border-radius: 50%;
    background: var(--emotion-ring);
    box-shadow:
        0 0 0 1px rgba(255, 255, 255, 0.05),
        0 18px 44px rgba(0, 0, 0, 0.28);
    transition: background 280ms ease;
}

.compass-core {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 18px;
    border: 1px solid var(--line);
    border-radius: 50%;
    background: #0D1016;
    text-align: center;
}

.compass-core span {
    color: var(--muted);
    font-size: 11px;
    line-height: 1.2;
    font-weight: 700;
    text-transform: uppercase;
}

.compass-core strong {
    max-width: 132px;
    margin-top: 8px;
    color: var(--text);
    font-family: "Segoe UI Variable Display", "Segoe UI", Arial, sans-serif;
    font-size: 34px;
    line-height: 1;
    overflow-wrap: anywhere;
}

.reading-copy h2 {
    max-width: 650px;
    margin: 0 0 13px;
    color: var(--text);
    font-family: "Segoe UI Variable Display", "Segoe UI", Arial, sans-serif;
    font-size: 38px;
    line-height: 1.06;
    font-weight: 700;
}

.reading-copy > p {
    max-width: 610px;
    margin: 0;
    color: #B8C0CE;
    font-size: 15px;
    line-height: 1.5;
}

.next-step {
    max-width: 610px;
    margin-top: 18px;
    padding: 2px 0 2px 17px;
    border-left: 3px solid var(--primary-color, var(--line-strong));
}

.next-step span {
    color: var(--muted);
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
}

.next-step p {
    margin: 7px 0 0;
    color: var(--text);
    font-size: 14px;
    line-height: 1.5;
}

.is-empty .reading-copy h2 {
    color: #D9DEE7;
}

.profile-content {
    height: 100%;
    min-height: 0;
    display: flex;
    flex-direction: column;
}

.profile-heading {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 20px;
    margin-bottom: 16px;
}

.profile-heading p {
    max-width: 190px;
    margin: 0;
    color: var(--faint);
    font-size: 12px;
    line-height: 1.4;
    text-align: right;
}

.profile-list {
    display: grid;
    gap: 9px;
}

.profile-row {
    display: grid;
    gap: 5px;
}

.profile-meta {
    display: grid;
    grid-template-columns: 10px 1fr auto;
    align-items: center;
    gap: 9px;
    color: #DDE2EA;
    font-size: 13px;
}

.profile-meta strong {
    min-width: 32px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
    text-align: right;
}

.emotion-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--dot-color);
}

.profile-track {
    height: 5px;
    overflow: hidden;
    border-radius: 3px;
    background: #252B35;
}

.profile-track span {
    display: block;
    height: 100%;
    min-width: 0;
    border-radius: inherit;
    background: var(--bar-color);
    transition: width 360ms ease;
}

button:focus-visible {
    outline: 2px solid #596272 !important;
    outline-offset: 2px !important;
}

@media (max-width: 1080px) {
    .page-shell {
        padding-inline: 14px;
    }

    .composer-grid,
    .results-row {
        flex-wrap: nowrap !important;
    }

    .composer-panel {
        flex-basis: clamp(250px, 32vh, 300px);
    }

    .composer-side {
        min-width: 172px !important;
    }

    .composer-actions {
        min-width: 142px !important;
    }

    #message-input {
        min-width: 240px !important;
    }

    .reading-panel {
        min-width: 0 !important;
    }

    .profile-panel {
        min-width: 240px !important;
    }

    .reading-state {
        grid-template-columns: 145px minmax(0, 1fr);
        gap: 20px;
    }

    .compass-ring {
        width: min(18vh, 150px);
        padding: 11px;
    }

    .compass-core {
        padding: 11px;
    }

    .compass-core strong {
        font-size: 25px;
    }

    .reading-copy h2 {
        font-size: 30px;
    }

    .profile-heading p {
        display: none;
    }
}

@media (max-height: 820px) and (min-width: 721px) {
    .page-shell {
        padding-bottom: 10px;
        gap: 8px !important;
    }

    .app-header {
        height: 56px;
        min-height: 56px;
        flex-basis: 56px;
    }

    .brand-copy p,
    .composer-heading > p {
        display: none !important;
    }

    .composer-panel {
        flex-basis: min(255px, 35vh);
        padding: 14px !important;
        gap: 9px !important;
    }

    .composer-heading h2,
    .profile-heading h2 {
        font-size: 20px;
    }

    .composer-grid {
        gap: 10px !important;
    }

    .sample-stack {
        gap: 3px !important;
    }

    .sample-button {
        min-height: 29px !important;
        font-size: 11px !important;
    }

    #message-input textarea {
        padding: 13px !important;
        font-size: 15px !important;
        line-height: 1.5 !important;
    }

    #read-button {
        min-height: 58px !important;
    }

    .reading-panel,
    .profile-panel {
        padding: 15px !important;
    }

    .reading-copy h2 {
        margin-bottom: 8px;
        font-size: 28px;
    }

    .reading-copy > p {
        font-size: 13px;
    }

    .next-step {
        margin-top: 10px;
    }

    .profile-heading {
        margin-bottom: 10px;
    }

    .profile-list {
        gap: 6px;
    }

    .profile-row {
        gap: 3px;
    }

}

@media (max-width: 720px) {
    .page-shell {
        padding: 0 8px 8px;
        gap: 8px !important;
    }

    .app-header {
        height: 54px;
        min-height: 54px;
        flex-basis: 54px;
    }

    .header-note,
    .brand-copy p,
    .composer-heading > p,
    .action-note {
        display: none !important;
    }

    .brand-mark {
        width: 30px;
        height: 30px;
        flex-basis: 30px;
    }

    .brand-copy h1 {
        font-size: 19px;
    }

    .composer-panel {
        flex: 0 0 min(280px, 38vh);
        padding: 10px !important;
        gap: 7px !important;
    }

    .composer-heading {
        align-items: flex-start;
    }

    .section-label {
        margin-bottom: 4px;
        font-size: 9px;
    }

    .composer-heading h2 {
        font-size: 18px;
    }

    .composer-grid {
        flex-direction: column !important;
        flex-wrap: nowrap !important;
        gap: 7px !important;
    }

    .composer-side,
    .composer-actions {
        width: 100%;
        min-width: 0 !important;
    }

    .composer-side {
        height: auto;
        flex: 0 0 auto !important;
        display: block !important;
        justify-content: flex-start;
    }

    .mini-label {
        display: none;
    }

    .sample-stack {
        display: grid !important;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 4px !important;
    }

    .sample-button {
        min-width: 0 !important;
        min-height: 31px !important;
        padding: 4px !important;
        justify-content: center !important;
        font-size: 10px !important;
        text-align: center !important;
        white-space: nowrap;
    }

    #message-input {
        width: 100%;
        height: auto !important;
        min-width: 0 !important;
        flex: 1 1 auto !important;
    }

    #message-input textarea {
        padding: 11px !important;
        font-size: 14px !important;
        line-height: 1.45 !important;
    }

    .composer-actions {
        height: 40px;
        min-height: 40px !important;
        flex: 0 0 40px !important;
        flex-direction: row !important;
        gap: 7px !important;
    }

    #read-button {
        height: 40px;
        min-height: 40px !important;
        flex: 1 1 auto !important;
        font-size: 14px !important;
    }

    #clear-button {
        height: 40px;
        min-height: 40px !important;
        flex: 0 0 78px !important;
    }

    .results-row {
        flex-wrap: nowrap !important;
    }

    .profile-panel {
        display: none !important;
    }

    .reading-panel {
        width: 100%;
        min-width: 0 !important;
        padding: 12px !important;
    }

    .reading-state {
        grid-template-columns: 102px minmax(0, 1fr);
        gap: 12px;
    }

    .compass-ring {
        width: 98px;
        padding: 7px;
    }

    .compass-core {
        padding: 6px;
    }

    .compass-core span {
        font-size: 8px;
    }

    .compass-core strong {
        margin-top: 4px;
        font-size: 17px;
    }

    .reading-copy h2 {
        margin-bottom: 7px;
        font-size: 21px;
        line-height: 1.1;
    }

    .reading-copy > p {
        font-size: 12px;
        line-height: 1.4;
    }

    .next-step {
        margin-top: 9px;
        padding-left: 10px;
    }

    .next-step span {
        font-size: 8px;
    }

    .next-step p {
        margin-top: 4px;
        font-size: 11px;
        line-height: 1.35;
    }
}

@media (prefers-reduced-motion: reduce) {
    .compass-ring,
    .profile-track span {
        transition: none;
    }
}
"""


with gr.Blocks(
    title=APP_TITLE,
    fill_width=True,
    fill_height=True,
) as demo:
    with gr.Column(elem_classes=["page-shell"]):
        gr.HTML(
            """
            <header class="app-header">
                <div class="brand">
                    <div class="brand-mark" aria-hidden="true"><span></span></div>
                    <div class="brand-copy">
                        <h1>Feeling Compass</h1>
                        <p>Find the emotional direction behind the words.</p>
                    </div>
                </div>
                <div class="header-note"><span></span>Ready for a new reading</div>
            </header>
            """,
            elem_id="app-header",
        )

        with gr.Column(elem_classes=["surface-panel", "composer-panel"]):
            gr.HTML(
                """
                <div class="composer-heading">
                    <div>
                        <span class="section-label">Words to read</span>
                        <h2>What is being expressed?</h2>
                    </div>
                </div>
                """,
                elem_id="composer-heading",
            )

            with gr.Row(elem_classes=["composer-grid"]):
                with gr.Column(
                    scale=2,
                    min_width=210,
                    elem_classes=["composer-side"],
                ):
                    with gr.Column(elem_classes=["sample-stack"]):
                        gr.Markdown("Try a sample", elem_classes=["mini-label"])
                        tense_sample = gr.Button(
                            "Tense",
                            variant="secondary",
                            elem_classes=["sample-button"],
                        )
                        bright_sample = gr.Button(
                            "Happy",
                            variant="secondary",
                            elem_classes=["sample-button"],
                        )
                        uncertain_sample = gr.Button(
                            "Worried",
                            variant="secondary",
                            elem_classes=["sample-button"],
                        )

                text_input = gr.Textbox(
                    label="Writing to read",
                    show_label=False,
                    placeholder="Paste a message, note, review, or journal entry here...",
                    lines=8,
                    max_lines=12,
                    autofocus=False,
                    container=False,
                    elem_id="message-input",
                    scale=7,
                    min_width=360,
                )

                with gr.Column(
                    scale=2,
                    min_width=180,
                    elem_classes=["composer-actions"],
                ):
                    read_button = gr.Button(
                        "Read the tone",
                        variant="primary",
                        elem_id="read-button",
                    )
                    clear_button = gr.Button(
                        "Clear",
                        variant="secondary",
                        elem_id="clear-button",
                    )
                    gr.Markdown(
                        "Leave out names or private details you do not want to share.",
                        elem_classes=["action-note"],
                    )

        with gr.Row(elem_classes=["results-row"]):
            with gr.Column(
                scale=7,
                min_width=520,
                elem_classes=["surface-panel", "reading-panel"],
            ):
                reading_output = gr.HTML(
                    value=render_waiting_reading(),
                    elem_id="reading-output",
                )

            with gr.Column(
                scale=4,
                min_width=360,
                elem_classes=["surface-panel", "profile-panel"],
            ):
                profile_output = gr.HTML(
                    value=render_profile(),
                    elem_id="profile-output",
                )

    read_button.click(
        fn=analyze_text,
        inputs=[text_input],
        outputs=[reading_output, profile_output],
        show_progress="minimal",
    )

    clear_button.click(
        fn=clear_app,
        inputs=None,
        outputs=[text_input, reading_output, profile_output],
        queue=False,
    )

    tense_sample.click(
        fn=lambda: SAMPLES["tense"],
        inputs=None,
        outputs=text_input,
        queue=False,
    )

    bright_sample.click(
        fn=lambda: SAMPLES["bright"],
        inputs=None,
        outputs=text_input,
        queue=False,
    )

    uncertain_sample.click(
        fn=lambda: SAMPLES["uncertain"],
        inputs=None,
        outputs=text_input,
        queue=False,
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=8).launch(
        theme=THEME,
        css=CSS,
        show_error=False,
        footer_links=[],
        enable_monitoring=False,
    )
