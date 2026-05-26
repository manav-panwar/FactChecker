import fitz  # pymupdf
import json
from groq import Groq

MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = (
    "You are a claim extractor. Extract only specific verifiable claims "
    "(stats, percentages, dates, financial figures, technical specs). "
    "Skip opinions and marketing language. "
    "Return ONLY a valid JSON array: "
    '[{"claim": "...", "type": "stat|date|financial|technical"}]'
)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    doc.close()
    return text


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]).strip()
    return raw


def extract_claims(text: str, client: Groq) -> tuple[list[dict], dict]:
    """
    Returns (claims, usage) where usage = {"input": int, "output": int}.
    json.JSONDecodeError propagates to caller (app.py shows st.error).
    """
    truncated = text[:12000]

    response = client.chat.completions.create(
        model=MODEL,
        reasoning_effort="low",
        max_completion_tokens=4000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract claims from this text:\n\n{truncated}"},
        ],
    )

    usage = {
        "input": response.usage.prompt_tokens if response.usage else 0,
        "output": response.usage.completion_tokens if response.usage else 0,
    }

    raw = response.choices[0].message.content.strip()
    cleaned = _strip_fences(raw)
    return json.loads(cleaned), usage
