# HackMIT-Team-2025

HackMIT 2025 project repository.

## Overview
A tool for transforming medical transcripts into:
- **Readable notes** (concise summaries, key points, action items)
- **Structured JSON** (tasks, owners, dates, entities)
- **Web Interface** (user-friendly frontend for easy access)

## Features
- 🏥 AI-powered SOAP note generation from clinical transcripts
- 🌐 Modern web interface with drag-and-drop file upload
- 📋 Copy to clipboard and download functionality
- 🤖 Support for multiple AI providers (OpenAI, Anthropic, Gemini)
- 📱 Responsive design for mobile and desktop

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/amoveablefeastym/HackMIT-Team-2025.git
cd HackMIT-Team-2025
```

### 2. Set up Python environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install anthropic langextract python-dotenv tenacity openai flask flask-cors
```

### 3. Configure API keys
```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required API keys (get at least one):
- **OpenAI:** https://platform.openai.com/api-keys (recommended)
- **Anthropic:** https://console.anthropic.com/settings/keys
- **Gemini:** https://aistudio.google.com/app/apikey

### 4. Run the application

#### Option A: Web Interface (Recommended)

Start the backend server:
```bash
python app.py
```

Then open `frontend/index.html` in your browser, or serve it:
```bash
cd frontend
python -m http.server 8000
```
Visit: http://localhost:8000

#### Option B: Command Line

Generate SOAP notes:
```bash
cd transcript-to-notes
python transcript_to_notes_openai.py
```

Generate structured data:
```bash
cd transcript-to-structured
python transcript_to_structured.py
```

## Project Structure

```
HackMIT-Team-2025/
├── frontend/              # Web interface
│   ├── index.html        # Main page
│   ├── styles.css        # Styling
│   ├── script.js         # Frontend logic
│   └── README.md         # Frontend docs
├── transcript-to-notes/   # SOAP note generation
│   ├── input.txt         # Sample transcript
│   ├── transcript_to_notes.py          # Anthropic version
│   ├── transcript_to_notes_openai.py   # OpenAI version
│   └── unstructured_clinical_notes.txt # Output
├── transcript-to-structured/  # Structured data extraction
│   ├── input.txt         # Sample transcript
│   ├── transcript_to_structured.py
│   └── visualization.html
├── app.py                # Flask backend API
├── .env.example          # API key template
└── README.md            # This file
```

## Usage

### Web Interface

1. **Upload or paste** a medical transcript
2. **Click "Generate SOAP Note"**
3. **View, copy, or download** the result

### Command Line

```bash
# Using OpenAI (recommended)
export OPENAI_API_KEY="your-key-here"
python transcript-to-notes/transcript_to_notes_openai.py

# Using Anthropic
export ANTHROPIC_API_KEY="your-key-here"
python transcript-to-notes/transcript_to_notes.py

# Using Gemini
export LANGEXTRACT_API_KEY="your-key-here"
python transcript-to-structured/transcript_to_structured.py
```

## API Documentation

### Generate SOAP Note
```http
POST /api/generate-soap
Content-Type: application/json

{
  "transcript": "Doctor: Hello... Patient: Hi..."
}
```

Response:
```json
{
  "success": true,
  "soap_note": "**Subjective:**\n...",
  "model": "gpt-4o-mini"
}
```

## Example

**Input:**
```
Alice: Let's finish slides by Monday.
Bob: I'll handle the graphs.
```

**SOAP Note Output:**
```
Subjective: Patient reports difficulty sleeping...
Objective: Virtual appointment conducted...
Assessment: Insomnia secondary to stress...
Plan: Sleep hygiene education...
```

**Structured JSON Output:**
```json
{
  "tasks": [
    {"text": "Finish slides", "owner": "Alice", "due": "Monday"},
    {"text": "Prepare graphs", "owner": "Bob"}
  ]
}
```

## Technologies Used

- **Backend:** Python, Flask
- **AI Models:** OpenAI GPT-4o-mini, Anthropic Claude, Google Gemini
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Libraries:** LangExtract, python-dotenv, tenacity

## Privacy & Security

⚠️ **Important:** This is a demo application for educational purposes.

For production use with real patient data:
- Implement HIPAA-compliant data handling
- Use encrypted connections (HTTPS)
- Add authentication and authorization
- Implement audit logging
- Follow all healthcare data regulations
- Never commit API keys to version control

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub.

---

**HackMIT 2025** | Built with ❤️ for better healthcare documentation
