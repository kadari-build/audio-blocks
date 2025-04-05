# Audio Blocks

An interactive AI voice assistant with dynamic particle visualization. This application combines voice interaction with beautiful visual feedback, creating an engaging and responsive user interface.

## Features

- Real-time voice interaction with AI
- Dynamic dual-particle visualization that responds to audio input
- Light/dark theme support
- High-quality text-to-speech synthesis
- Beautiful, modern UI with smooth animations
- Real-time audio visualization
- Responsive design

## Technologies Used

- Python (Flask) for backend
- Google Cloud Text-to-Speech API
- Google Generative AI (Gemini) for chat
- Web Speech API for voice recognition
- HTML5 Canvas for particle visualization
- Modern JavaScript for real-time interactions

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/audio-blocks.git
cd audio-blocks
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your API keys:
```
GOOGLE_API_KEY=your_google_api_key
```

4. Start the server:
```bash
python server.py
```

5. Open your browser and navigate to `http://localhost:7777`

## Usage

1. Click the microphone icon or press space to start speaking
2. Watch the particle visualization respond to your voice
3. The AI will respond both visually and audibly
4. Use the theme toggle in the top-right to switch between light and dark modes

## Requirements

- Python 3.8+
- Modern web browser with Web Speech API support
- Google Cloud API key with Text-to-Speech and Gemini enabled

## License

MIT License - See LICENSE file for details 