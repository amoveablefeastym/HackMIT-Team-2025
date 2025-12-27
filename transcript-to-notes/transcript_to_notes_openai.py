#!/usr/bin/env python3
import os
from pathlib import Path
from openai import OpenAI

PROMPT_TEXT = (
    "Summarize the following outpatient visit as a SOAP note (≤120 words). "
    "Keep it strictly faithful to the text; do not add facts. "
    "Format sections: Subjective, Objective, Assessment, Plan."
)

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def main():
    # ---- file paths ----
    transcript_path = Path("input.txt")
    output_path = Path("unstructured_clinical_notes.txt")

    # ---- sanity checks ----
    if not transcript_path.exists():
        raise FileNotFoundError(f"Missing transcript file: {transcript_path.resolve()}")

    transcript_text = read_text(transcript_path)

    # ---- OpenAI client ----
    # Requires env var: OPENAI_API_KEY
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Call OpenAI GPT-4
    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-4o" for more powerful model
        max_tokens=512,
        temperature=0,  # deterministic, avoids adding facts
        messages=[
            {
                "role": "system",
                "content": "You are a medical assistant that creates concise SOAP notes from clinical transcripts."
            },
            {
                "role": "user",
                "content": f"{PROMPT_TEXT}\n\nTranscript:\n\n{transcript_text}"
            }
        ],
    )

    # Extract text from the response and write to file
    output_text = resp.choices[0].message.content.strip()

    output_path.write_text(output_text + "\n", encoding="utf-8")
    print(f"Wrote SOAP note to: {output_path.resolve()}")

if __name__ == "__main__":
    main()
