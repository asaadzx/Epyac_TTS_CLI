# 🎯 Epyac TTS CLI - Text-to-Speech Generator 🎙️

A powerful command-line interface for generating high-quality speech using the Bark Text-to-Speech model. Convert your text into natural-sounding voice with multiple speaker options!

## ✨ Features

- 🚀 Easy-to-use command-line interface
- 🎭 Multiple voice presets available
- 📊 Real-time progress tracking
- 💪 CUDA support for faster generation
- 🎨 Clean and intuitive user interface
- 🌐 Multi-language support:
    - English
    - German
    - Spanish
    - French
    - Hindi
    - Italian
    - Japanese
    - Korean
    - Polish
    - Portuguese
    - Russian
    - Turkish
    - Chinese

## 🖥️ CLI Interface

```
========================================
 /$$$$$$$$                                                     
| $$_____/                                                     
| $$        /$$$$$$  /$$   /$$  /$$$$$$   /$$$$$$$             
| $$$$$    /$$__  $$| $$  | $$ |____  $$ /$$_____/             
| $$__/   | $$  \ $$| $$  | $$  /$$$$$$$| $$                   
| $$      | $$  | $$| $$  | $$ /$$__  $$| $$                   
| $$$$$$$$| $$$$$$$/|  $$$$$$$|  $$$$$$$|  $$$$$$$             
|________/| $$____/  \____  $$ \_______/ \_______/             
          | $$       /$$  | $$                                 
          | $$      |  $$$$$$/                                 
          |__/       \______/                                  
 /$$$$$$$$ /$$$$$$$$ /$$$$$$         /$$$$$$  /$$       /$$$$$$
|__  $$__/|__  $$__//$$__  $$       /$$__  $$| $$      |_  $$_/
   | $$      | $$  | $$  \__/      | $$  \__/| $$        | $$  
   | $$      | $$  |  $$$$$$       | $$      | $$        | $$  
   | $$      | $$   \____  $$      | $$      | $$        | $$  
   | $$      | $$   /$$  \ $$      | $$    $$| $$        | $$  
   | $$      | $$  |  $$$$$$/      |  $$$$$$/| $$$$$$$$ /$$$$$$
   |__/      |__/   \______/        \______/ |________/|______/
========================================
```

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/asaadzx/Epyac_TTS_CLI

# Navigate to project directory
cd Epyac_TTS_CLI

# Install required packages
pip install requirements-detector
```

## 🚀 Usage

1. Run the script:
```bash
python tts_cli.py
```

2. Follow the interactive prompts:
        - Enter your text or load from file
        - Choose language and voice preset
        - Specify output file location

## 🎧 Audio Samples

Here are some examples of generated audio:

| Text | Voice Preset | Audio |
|------|-------------|-------|
| "Hello World!" | v2/en_speaker_3 | [Listen](samples/hello_world.wav) |
| "Welcome to TTS" | v2/en_speaker_6 | [Listen](samples/welcome.wav) |
| "Test" | v2/en_speaker_2 | [Listen](samples/test.wav) |
<audio controls autoplay>
  <source src="samples/test.mp3" type="audio/mpeg">
Your browser does not support the audio element.
</audio>
## 🔧 System Requirements

- Python 3.7+
- CUDA-capable GPU (optional, but recommended)
- Minimum 4GB RAM
- Storage space for models

## 👥 Contributing

Feel free to:
- Open issues
- Submit pull requests
- Suggest improvements
- Share your experience

## 🙏 Acknowledgments

- [Bark TTS Model](https://github.com/suno-ai/bark)
- [Hugging Face Transformers](https://huggingface.co/transformers)
