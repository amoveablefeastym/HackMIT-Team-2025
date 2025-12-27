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

## Usage

# Generate notes
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-xxxx..."
python transcript-to-notes.py

# Generate structured JSON
pip install langextract python-dotenv tenacity
cat > .env << 'EOF'
# REQUIRED for LangExtract when using Gemini backend
LANGEXTRACT_API_KEY=YOUR_GEMINI_API_KEY
python transcript-to-structured.py

## Example
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
