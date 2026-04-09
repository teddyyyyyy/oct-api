import json
import os
import urllib.error
import urllib.request

OLLAMA_VERSION = os.getenv("OLLAMA_VERSION", "0.20.4")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "30"))


def generate_report(label: str, probs: dict, drift: dict) -> str:
    """
    Rule-based fallback report.
    Safe, deterministic, and always available.
    """
    if not probs or not isinstance(probs, dict):
        return "ERROR: Invalid probability data"

    top = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    confidence_lines = []
    for i, (cls, prob) in enumerate(top[:3]):
        confidence_lines.append(f"{i+1}. {cls}: {prob:.3f}")
    confidence_text = "\n".join(confidence_lines)

    drift_note = ""
    if drift and drift.get("alert"):
        drift_note = (
            f"\n⚠️ Data shift alert: {drift['alert']} "
            f"(score={drift.get('score', 0):.2f})"
        )

    return (
        "OCT AI Preliminary Report\n"
        "=========================\n"
        f"Ollama version: {OLLAMA_VERSION}\n"
        f"Predicted class: {label}\n"
        f"Confidence ranking:\n{confidence_text}\n\n"
        "Clinical note:\n"
        "- This output is for research/triage support only and is NOT a final diagnosis.\n"
        "- Please correlate with symptoms, fundus, and physician assessment.\n"
        f"{drift_note}\n"
    )


def build_oct_prompt(label: str, probs: dict, drift: dict) -> str:
    """
    Build a grounded prompt from classifier output only.
    """
    top = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top3 = "\n".join([f"{i+1}. {cls}: {prob:.4f}" for i, (cls, prob) in enumerate(top[:3])])

    drift_text = "No drift alert."
    if drift and drift.get("alert"):
        drift_text = (
            f"Drift alert detected: {drift['alert']} "
            f"(score={drift.get('score', 0):.2f})."
        )

    return f"""
You are assisting with a research OCT triage report.

Model metadata:
- Runtime: Ollama v{OLLAMA_VERSION}
- Classifier output label: {label}

Classifier confidence ranking:
{top3}

Drift monitoring:
- {drift_text}

Instructions:
- Write a concise OCT report in plain clinical English.
- Use ONLY the classifier output and drift information above.
- Do NOT invent image findings that were not provided.
- Do NOT claim a confirmed diagnosis.
- State that this is research/triage support only.
- End with a brief recommendation for clinician correlation.

Output format:
Impression:
<2-4 sentences>

Recommendation:
<1-2 sentences>
""".strip()


def generate_llm_report(label: str, probs: dict, drift: dict) -> str:
    """
    Call local Ollama /api/generate and return the model text.
    Raises exception if Ollama is unavailable or returns invalid output.
    """
    prompt = build_oct_prompt(label, probs, drift)

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": (
            "You are a careful medical AI writing assistant. "
            "Be concise, cautious, and grounded. "
            "Never overstate certainty."
        ),
        "stream": False,
    }

    req = urllib.request.Request(
        url=f"{OLLAMA_HOST}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
        raw = resp.read().decode("utf-8")
        data = json.loads(raw)

    text = data.get("response", "").strip()
    if not text:
        raise RuntimeError("Empty response from Ollama")

    return text


def generate_report_with_fallback(label: str, probs: dict, drift: dict) -> tuple[str, str]:
    """
    Returns:
        report_text, report_source
    """
    try:
        report = generate_llm_report(label, probs, drift)
        return report, "llm"
    except Exception:
        fallback = generate_report(label, probs, drift)
        return fallback, "template"