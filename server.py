"""
AI Voice Assistant Server
------------------------
This Flask server handles the backend processing for the AI Voice Assistant.
It integrates with Google's Gemini API for chat functionality and Google Cloud
Text-to-Speech for voice synthesis.

Key Features:
- Chat session management
- Voice synthesis
- API integration
- CORS handling for local development
- Async task processing
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import google.generativeai as genai
from google.cloud import texttospeech_v1
import os
from dotenv import load_dotenv
import time
import io
import asyncio
import threading
from queue import Queue
import re

# Initialize Flask app with CORS support
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:7777", "http://127.0.0.1:7777"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Session-ID"]
    }
})

# Load environment variables from .env file
load_dotenv()

# Initialize Text-to-Speech client with API key
tts_client = texttospeech_v1.TextToSpeechClient(
    client_options={"api_key": os.getenv('GOOGLE_API_KEY')}
)

# Task processing queue and patterns
task_queue = Queue()
PRINCIPAL_COMMANDS = {
    r"(?i)as principal,?\s*(approve|deny|review)\s+(.+)": "Administrative",
    r"(?i)as principal,?\s*(schedule|arrange)\s+(.+)": "Scheduling",
    r"(?i)as principal,?\s*(contact|email|call)\s+(.+)": "Communication",
    r"(?i)as principal,?\s*(implement|establish)\s+(.+)": "Policy",
    r"(?i)as principal,?\s*(evaluate|assess)\s+(.+)": "Evaluation",
    r"(?i)as principal,?\s*(authorize|permit)\s+(.+)": "Authorization"
}

def process_tasks():
    """Background task processor"""
    while True:
        try:
            session_id, task = task_queue.get()
            if session_id in chat_sessions:
                session = chat_sessions[session_id]
                if 'tasks' not in session:
                    session['tasks'] = []
                session['tasks'].append(task)
                print(f"Processed task for session {session_id}: {task}")
            task_queue.task_done()
        except Exception as e:
            print(f"Error processing task: {e}")

# Start task processing thread
task_thread = threading.Thread(target=process_tasks, daemon=True)
task_thread.start()

def check_principal_command(text):
    """
    Check if text contains a principal command and categorize it
    
    Args:
        text (str): The message text to check
        
    Returns:
        tuple: (command_type, command_content) or (None, None) if no command found
    """
    for pattern, category in PRINCIPAL_COMMANDS.items():
        match = re.match(pattern, text)
        if match:
            action = match.group(1)
            content = match.group(2)
            return category, f"{action.capitalize()}: {content}"
    return None, None

def clean_text_for_speech(text):
    """
    Clean text by removing markdown formatting and other symbols that shouldn't be spoken.
    
    Args:
        text (str): The text to clean
        
    Returns:
        str: Cleaned text ready for speech synthesis
    """
    # Remove markdown bold/italic
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.+?)\*', r'\1', text)      # Italic
    text = re.sub(r'\_(.+?)\_', r'\1', text)      # Underscore emphasis
    
    # Remove markdown links
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    
    # Remove markdown code blocks and inline code
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    
    # Remove markdown headers
    text = re.sub(r'#{1,6}\s+', '', text)
    
    # Remove bullet points and numbering
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def generate_speech(text):
    """
    Generate high-quality speech from text using Google Cloud TTS.
    
    Args:
        text (str): The text to convert to speech
        
    Returns:
        bytes: The audio content in MP3 format
        
    Raises:
        Exception: If speech generation fails
    """
    try:
        # Clean the text before synthesis
        cleaned_text = clean_text_for_speech(text)
        
        synthesis_input = texttospeech_v1.SynthesisInput(text=cleaned_text)
        
        # Configure voice parameters for high quality output
        voice = texttospeech_v1.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Studio-O",  # Studio voice for highest quality
            ssml_gender=texttospeech_v1.SsmlVoiceGender.FEMALE
        )

        # Configure audio parameters for optimal quality
        audio_config = texttospeech_v1.AudioConfig(
            audio_encoding=texttospeech_v1.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0,
            sample_rate_hertz=24000,
            effects_profile_id=["headphone-class-device"]
        )

        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        return response.audio_content
    except Exception as e:
        raise Exception(f"Failed to generate speech: {str(e)}")

def init_gemini(api_key):
    """
    Initialize Gemini chat model with specific configuration.
    
    Args:
        api_key (str): The Gemini API key
        
    Returns:
        genai.ChatSession: Initialized chat session
        
    Raises:
        Exception: If initialization fails
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash',
            generation_config={
                "temperature": 0.7,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 150,
            },
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        )

        # Initialize chat with professional assistant context
        chat = model.start_chat(history=[
            {
                "role": "user",
                "parts": ["I want you to act as a professional AI assistant like Siri. Be helpful, efficient, and focused on tasks. When I ask you to take notes or create documents, maintain those in a list. Keep responses clear and professional."]
            },
            {
                "role": "model",
                "parts": ["I'll assist you professionally with your tasks. How can I help you today?"]
            }
        ])
        
        return chat
    except Exception as e:
        raise Exception(f"Failed to initialize Gemini: {str(e)}")

# Global store for active chat sessions
chat_sessions = {}

# Route handlers
@app.route('/')
def root():
    """Serve the main application page"""
    return app.send_static_file('index.html')

@app.route('/test', methods=['GET', 'OPTIONS'])
def test_connection():
    """Test endpoint for checking server connectivity"""
    if request.method == 'OPTIONS':
        return '', 204
    try:
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/init', methods=['POST', 'OPTIONS'])
def init_session():
    """Initialize a new chat session"""
    if request.method == 'OPTIONS':
        return '', 204
    try:
        # Use API key from environment variables
        api_key = os.getenv('GOOGLE_GENERATIVE_LANGUAGE_API_KEY')
        
        if not api_key:
            return jsonify({'error': 'API key not found in environment variables'}), 400

        # Initialize Gemini chat
        try:
            chat = init_gemini(api_key)
        except Exception as e:
            return jsonify({'error': str(e)}), 400

        # Create new session
        session_id = str(time.time())
        chat_sessions[session_id] = {
            'chat': chat
        }
        
        return jsonify({
            'session_id': session_id,
            'status': 'initialized'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/synthesize', methods=['POST', 'OPTIONS'])
def synthesize_speech():
    """Generate speech from text"""
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.json
        text = data.get('text')
        if not text:
            return jsonify({'error': 'Text is required'}), 400

        audio_content = generate_speech(text)
        
        return send_file(
            io.BytesIO(audio_content),
            mimetype='audio/mp3',
            as_attachment=True,
            download_name='response.mp3'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Handle chat messages and generate responses"""
    if request.method == 'OPTIONS':
        return '', 204
    try:
        # Validate session
        session_id = request.headers.get('Session-ID')
        if not session_id or session_id not in chat_sessions:
            return jsonify({'error': 'Invalid session'}), 400

        # Get message
        data = request.json
        message = data.get('message')
        if not message:
            return jsonify({'error': 'Message is required'}), 400

        # Check for principal commands
        command_type, command_content = check_principal_command(message)
        task_added = False
        
        if command_type:
            # Add task to processing queue
            task = {
                'type': command_type,
                'content': command_content,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'pending'
            }
            task_queue.put((session_id, task))
            task_added = True

        # Get chat session
        session = chat_sessions[session_id]
        chat = session['chat']

        # Generate response
        response = chat.send_message(message)
        
        # Update conversation history
        if 'history' not in session:
            session['history'] = []
        
        session['history'].append({
            'user': message,
            'ai': response.text
        })

        # Generate speech for response
        try:
            audio_content = generate_speech(response.text)
            audio_base64 = audio_content.hex()
        except Exception as e:
            audio_base64 = None
            print(f"Speech synthesis error: {e}")
        
        return jsonify({
            'response': response.text,
            'audio': audio_base64,
            'status': 'success',
            'task_added': task_added,
            'command_type': command_type,
            'command_content': command_content
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting server on http://localhost:7777")
    print("Make sure to access the application through http://localhost:7777")
    app.run(host='localhost', port=7777, debug=True) 