#!/usr/bin/env python3
import base64
import os
from pathlib import Path
from anthropic import Anthropic

PROMPT_TEXT = (
    "Summarize the following outpatient visit as a SOAP note (≤120 words). "
    "Keep it strictly faithful to the text; do not add facts. "
    "Format sections: Subjective, Objective, Assessment, Plan."
)

def b64encode_pdf(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def main():
    # ---- file paths ----
    transcript_path = Path("input.txt")
    rubric_pdf_path = Path("SOAP Note Rubric.pdf")
    output_path = Path("unstructured_clinical_notes.txt")

    # ---- sanity checks ----
    if not transcript_path.exists():
        raise FileNotFoundError(f"Missing transcript file: {transcript_path.resolve()}")
    if not rubric_pdf_path.exists():
        raise FileNotFoundError(f"Missing rubric PDF: {rubric_pdf_path.resolve()}")

    transcript_text = read_text(transcript_path)
    rubric_pdf_b64 = b64encode_pdf(rubric_pdf_path)

    # ---- Anthropic client ----
    # Requires env var: ANTHROPIC_API_KEY
    client = Anthropic()

    # Construct content blocks:
    # 1) Provide the rubric PDF as a document block (so the model can consult it)
    # 2) Provide the exact instruction prompt
    # 3) Provide the raw transcript text
    content_blocks = [
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": rubric_pdf_b64,
            },
            # Optional: hint to cache this PDF for this request only (no reuse)
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": PROMPT_TEXT,
        },
        {
            "type": "text",
            "text": f"Transcript:\n\n{transcript_text}",
        },
    ]

    # Call Claude 3.5 Haiku (latest alias)
    # Note: model alias and PDF “document” blocks via Messages API are documented by Anthropic. :contentReference[oaicite:0]{index=0}
    resp = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=512,
        temperature=0,  # deterministic, avoids adding facts
        messages=[
            {
                "role": "user",
                "content": content_blocks,
            }
        ],
    )

    # Extract text from the response and write to file
    # (Claude returns a list of content blocks; we join any text parts)
    out_chunks = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            out_chunks.append(block.text)
    output_text = "\n".join(out_chunks).strip()

    output_path.write_text(output_text + "\n", encoding="utf-8")
    print(f"Wrote SOAP note to: {output_path.resolve()}")

if __name__ == "__main__":
    main()
