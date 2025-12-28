from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from pathlib import Path
from openai import OpenAI

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

PROMPT_TEXT = (
    "Summarize the following outpatient visit as a SOAP note (≤120 words). "
    "Keep it strictly faithful to the text; do not add facts. "
    "Format sections: Subjective, Objective, Assessment, Plan."
)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "API is running"})

@app.route('/api/generate-soap', methods=['POST'])
def generate_soap():
    """Generate SOAP note from transcript"""
    try:
        data = request.get_json()
        transcript = data.get('transcript', '').strip()
        
        if not transcript:
            return jsonify({"error": "Transcript text is required"}), 400
        
        # Check if OpenAI API key is set
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return jsonify({"error": "OpenAI API key not configured"}), 500
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Call OpenAI API
        response = client.chat.completions.create(
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
                    "content": f"{PROMPT_TEXT}\n\nTranscript:\n\n{transcript}"
                }
            ],
        )
        
        # Extract the generated text
        soap_note = response.choices[0].message.content.strip()
        
        return jsonify({
            "success": True,
            "soap_note": soap_note,
            "model": "gpt-4o-mini"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/sample-transcript', methods=['GET'])
def get_sample_transcript():
    """Get sample transcript"""
    try:
        sample_path = Path(__file__).parent / "transcript-to-notes" / "input.txt"
        if sample_path.exists():
            transcript = sample_path.read_text(encoding="utf-8", errors="ignore")
            return jsonify({
                "success": True,
                "transcript": transcript
            })
        else:
            return jsonify({
                "success": False,
                "error": "Sample transcript not found"
            }), 404
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the app
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
