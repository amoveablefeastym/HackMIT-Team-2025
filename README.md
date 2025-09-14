# HackMIT-Team-2025

HackMIT 2025 project repository.

## Overview
A tool for transforming transcripts into:
- **Readable notes** (concise summaries, key points, action items)
- **Structured JSON** (tasks, owners, dates, entities)

## Setup
```bash
git clone https://github.com/amoveablefeastym/HackMIT-Team-2025.git
cd HackMIT-Team-2025
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
Usage

# Generate notes
python transcript-to-notes/main.py examples/sample.txt

# Generate structured JSON
python transcript-to-structured/main.py examples/sample.txt
Example
Input:

Alice: Let's finish slides by Monday.
Bob: I'll handle the graphs.
Notes:

- Slides due Monday
- Bob will handle graphs
JSON:

{
  "tasks": [
    {"text": "Finish slides", "owner": "Alice", "due": "Monday"},
    {"text": "Prepare graphs", "owner": "Bob"}
  ]
}
