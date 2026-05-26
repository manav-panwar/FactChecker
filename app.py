import json
import time

import streamlit as st
from groq import Groq

from extractor import extract_claims, extract_text_from_pdf
from verifier import verify_claim

st.set_page_config(
    page_title="PDF Fact Checker",
    page_icon="🔍",
    layout="wide",
)

st.title("PDF Fact Checker")
st.caption("Upload a PDF — claims are extracted and verified against the web in real time.")


@st.cache_resource
def get_client() -> Groq:
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


client = get_client()

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()

    # ── Extraction ────────────────────────────────────────────────────────────
    with st.spinner("Extracting text and identifying claims..."):
        raw_text = extract_text_from_pdf(pdf_bytes)
        try:
            claims, extract_usage = extract_claims(raw_text, client)
        except Exception as e:
            st.error(f"Failed to parse claims from model response: {e}")
            st.stop()

    if not claims:
        st.warning("No verifiable claims were found in the document.")
        st.stop()

    total = len(claims)
    st.info(f"Found {total} claim{'s' if total != 1 else ''} — starting verification...")

    # ── Pre-allocate layout slots ─────────────────────────────────────────────
    summary_placeholder = st.empty()
    progress_bar = st.progress(0.0, text="Verifying claims...")
    card_placeholders = [st.empty() for _ in range(total)]

    counts: dict[str, int] = {"Verified": 0, "Inaccurate": 0, "Unverified": 0}
    total_input_tokens = extract_usage["input"]
    total_output_tokens = extract_usage["output"]

    def render_summary() -> None:
        summary_placeholder.markdown(
            f"**Live Results:** "
            f"✅ {counts['Verified']} Verified &nbsp;|&nbsp; "
            f"⚠️ {counts['Inaccurate']} Inaccurate &nbsp;|&nbsp; "
            f"❌ {counts['Unverified']} Unverified"
        )

    render_summary()

    # ── Verification loop ─────────────────────────────────────────────────────
    all_results: list[dict] = []

    for i, claim_obj in enumerate(claims):
        claim_text = claim_obj.get("claim", "")
        claim_type = claim_obj.get("type", "unknown")

        card_placeholders[i].info(f"Checking: _{claim_text}_")

        result, usage = verify_claim(claim_text, client)
        total_input_tokens += usage["input"]
        total_output_tokens += usage["output"]

        verdict = result.get("verdict", "Unverified")
        reason = result.get("reason", "")
        corrected = result.get("corrected_value")
        source = result.get("source")

        lines = [
            f"**Claim ({claim_type}):** {claim_text}",
            f"**Verdict:** {verdict}",
            f"**Reason:** {reason}",
        ]
        if corrected:
            lines.append(f"**Corrected Value:** {corrected}")
        if source:
            lines.append(f"**Source:** {source}")
        card_body = "\n\n".join(lines)

        if verdict == "Verified":
            card_placeholders[i].success(card_body)
        elif verdict == "Inaccurate":
            card_placeholders[i].warning(card_body)
        else:
            card_placeholders[i].error(card_body)

        counts[verdict] = counts.get(verdict, 0) + 1
        render_summary()
        progress_bar.progress((i + 1) / total, text=f"Verified {i + 1}/{total} claims...")

        all_results.append({"claim": claim_text, "type": claim_type, **result})

        time.sleep(0.5)

    # ── Done ──────────────────────────────────────────────────────────────────
    progress_bar.progress(1.0, text="All claims verified.")
