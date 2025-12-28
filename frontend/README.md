# HackMIT-Team-2025 Frontend

Web interface for the Clinical Transcript Analyzer.

## Features

- 📝 Upload or paste medical transcripts
- 🤖 AI-powered SOAP note generation
- 📋 Copy to clipboard
- 💾 Download as text file
- 📱 Responsive design
- 🎨 Modern, clean UI

## Quick Start

### Option 1: Static Frontend Only (Mock API)

Open `index.html` directly in your browser. This uses mock data and doesn't require a backend.

```bash
cd frontend
open index.html  # macOS
# or
xdg-open index.html  # Linux
# or just double-click the file in Windows
```

### Option 2: Full Stack with Backend API

1. **Install dependencies:**
```bash
pip install flask flask-cors
```

2. **Set up environment variables:**
Make sure your `.env` file has:
```bash
OPENAI_API_KEY=your-openai-api-key-here
```

3. **Run the Flask backend:**
```bash
python app.py
```

4. **Update frontend to use real API:**
Edit `script.js` and change the `USE_MOCK_API` flag:
```javascript
const USE_MOCK_API = false;  // Set to false to use real API
const API_URL = 'http://localhost:5000/api';
```

5. **Open the frontend:**
Open `frontend/index.html` in your browser, or serve it:
```bash
cd frontend
python -m http.server 8000
```
Then visit: http://localhost:8000

## Project Structure

```
frontend/
├── index.html       # Main HTML file
├── styles.css       # Styling
└── script.js        # JavaScript logic
```

## API Endpoints

### Health Check
```
GET /api/health
```

### Generate SOAP Note
```
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

## Technologies Used

- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Backend:** Flask (Python)
- **AI:** OpenAI GPT-4o-mini
- **Styling:** Custom CSS with modern design principles

## Privacy & Security Notice

⚠️ **Important:** This is a demo application. For production use with real patient data:

- Implement HIPAA-compliant data handling
- Use encrypted connections (HTTPS)
- Add authentication and authorization
- Implement audit logging
- Follow all healthcare data regulations

## Contributing

Feel free to open issues or submit pull requests!

## License

MIT License - See LICENSE file for details
