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
from google import genai
from google.genai import types
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

# Debug logging for API key
api_key = os.getenv('GOOGLE_API_KEY')
if api_key:
    print("API key loaded successfully")
    print(f"API key length: {len(api_key)}")
else:
    print("WARNING: API key not found in environment variables")

# Initialize Text-to-Speech client with API key
tts_client = texttospeech_v1.TextToSpeechClient(
    client_options={"api_key": api_key}
)

# Initialize Gemini
try:
    # Only configure the API key here, model initialization will happen in init_gemini
    genai.Client(api_key=api_key)

    print("Gemini API configured successfully")
except Exception as e:
    print(f"Failed to configure Gemini API: {str(e)}")

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

def generate_speech(text, agent_config):
    """
    Generate high-quality speech from text using Google Cloud TTS.
    
    Args:
        text (str): The text to convert to speech
        agent_config (dict): Configuration for the specific agent
        
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
            language_code=agent_config['voice']['language_code'],
            name=agent_config['voice']['name'],
            ssml_gender=agent_config['voice']['ssml_gender']
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

# Define available AI agents
AI_AGENTS = {
    'assistant': {
        'name': 'Director of Product',
        'model': 'gemini-2.0-flash',
        'system_prompt': "You are a Director of Product at our startup. Be professional, concise, and efficient. Make sure to be humanistic, and not too formal.",
        'voice': {
            'language_code': 'en-US',
            'name': 'en-US-Studio-O',
            'ssml_gender': texttospeech_v1.SsmlVoiceGender.FEMALE
        }
    },
    'creative': {
        'name': 'Director of Marketing',
        'model': 'gemini-2.0-flash',
        'system_prompt': "You are the Director of Marketing at our startup. Be professional, concise, and efficient. Make sure to be humanistic, and not too formal.",
        'voice': {
            'language_code': 'en-US',
            'name': 'en-US-Neural2-C',
            'ssml_gender': texttospeech_v1.SsmlVoiceGender.FEMALE
        }
    },
    'analyst': {
        'name': 'Director of Engineering',
        'model': 'gemini-2.0-flash',
        'system_prompt': "You are the Director of Engineering at our startup. Be professional, concise, and efficient. Make sure to be humanistic, and not too formal.",
        'voice': {
            'language_code': 'en-US',
            'name': 'en-US-Neural2-D',
            'ssml_gender': texttospeech_v1.SsmlVoiceGender.MALE
        }
    }
}

def init_gemini(api_key, agent_config):
    """
    Initialize Gemini chat model with specific configuration for an agent.
    
    Args:
        api_key (str): The Gemini API key
        agent_config (dict): Configuration for the specific agent
        
    Returns:
        genai.ChatSession: Initialized chat session
        
    Raises:
        Exception: If initialization fails
    """
    try:
        # Configure the API key
        client = genai.Client(api_key=api_key)

        chat_config = types.GenerateContentConfig(
            system_instruction=agent_config['system_prompt'],
            temperature=0.2,
        )

        chat = client.chats.create(
            model=agent_config['model'],
            config=chat_config,
            history=[]
        )   
        
        return chat
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg:
            raise Exception("Invalid or missing API key. Please check your GOOGLE_API_KEY environment variable.")
        elif "model" in error_msg.lower():
            raise Exception("Failed to initialize Gemini model. Please check if the model is available in your region.")
        else:
            raise Exception(f"Failed to initialize Gemini: {error_msg}")

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
    """Initialize a new chat session with multiple agents"""
    if request.method == 'OPTIONS':
        return '', 204
    try:
        # Use API key from environment variables
        api_key = os.getenv('GOOGLE_API_KEY')
        
        if not api_key:
            return jsonify({'error': 'API key not found in environment variables'}), 400

        # Initialize chat sessions for all agents
        session_id = str(time.time())
        chat_sessions[session_id] = {
            'agents': {}
        }
        
        # Initialize each agent
        for agent_id, agent_config in AI_AGENTS.items():
            try:
                chat = init_gemini(api_key, agent_config)
                chat_sessions[session_id]['agents'][agent_id] = {
                    'chat': chat,
                    'config': agent_config
                }
            except Exception as e:
                print(f"Failed to initialize agent {agent_id}: {str(e)}")
                continue
        
        if not chat_sessions[session_id]['agents']:
            return jsonify({'error': 'Failed to initialize any agents'}), 500
        
        return jsonify({
            'session_id': session_id,
            'agents': {k: v['config']['name'] for k, v in chat_sessions[session_id]['agents'].items()},
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

        audio_content = generate_speech(text, AI_AGENTS['assistant']['voice'])
        
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
    """Handle chat messages and generate responses from all agents"""
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

        # Get chat session
        session = chat_sessions[session_id]
        
        # Get responses from all agents
        responses = {}
        for agent_id, agent_data in session['agents'].items():
            try:
                # Generate response
                response = agent_data['chat'].send_message(message)
                response_text = response.text
                
                # Generate speech
                audio_content = generate_speech(response_text, agent_data['config'])
                audio_base64 = audio_content.hex()
                
                responses[agent_id] = {
                    'name': agent_data['config']['name'],
                    'response': response_text,
                    'audio': audio_base64
                }
                
                # Update conversation history
                if 'history' not in agent_data:
                    agent_data['history'] = []
                agent_data['history'].append({
                    'user': message,
                    'ai': response_text
                })
                
            except Exception as e:
                print(f"Error from agent {agent_id}: {str(e)}")
                responses[agent_id] = {
                    'name': agent_data['config']['name'],
                    'error': str(e)
                }
        
        return jsonify({
            'responses': responses,
            'status': 'success'
        })
    except Exception as e:
        print(f"Chat endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting server on http://localhost:7777")
    print("Make sure to access the application through http://localhost:7777")
    app.run(host='localhost', port=7777, debug=True) 