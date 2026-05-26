# PDF Fact Checker

A Streamlit application that extracts factual claims from uploaded PDF documents and verifies them against live web information using Groq's LLM with browser search.

## Features

- Extracts text directly from PDFs using PyMuPDF
- Automatically identifies factual claims
- Verifies claims using live web search
- Displays results in real time as verification completes
- Color-coded verdicts:
  - ✅ Verified
  - ⚠️ Inaccurate
  - ❌ Unverified
- Live summary statistics
- Export results as JSON

## Requirements

- Groq API Key

## Setup

Install dependencies:

```bash
uv sync
```

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_api_key_here"
```

Run the application:

```bash
uv run streamlit run app.py
```

## Tech Stack

- Streamlit
- Groq API
- PyMuPDF
- Python