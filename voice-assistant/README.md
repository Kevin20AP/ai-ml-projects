# Voice AI Assistant

A hands-free voice assistant that records speech, transcribes with **Whisper**, responds via **GPT-3.5**, and speaks back using **gTTS**.

## Features

- **Microphone recording** — 5-second audio capture via sounddevice
- **Speech-to-text** — OpenAI Whisper `base` model for transcription
- **Conversational AI** — GPT-3.5 with multi-turn memory
- **Text-to-speech** — gTTS with pygame audio playback
- **Concise responses** — tuned for spoken output (2–3 sentences)

## Stack

Whisper · OpenAI GPT-3.5 · gTTS · Pygame · Python

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

**Note:** Requires a working microphone. On macOS, grant microphone access when prompted.

## Run

```bash
python voice_assistant.py
```

Press Enter to record, type `quit` to exit.

## Pipeline

```
Microphone → Whisper (STT) → GPT-3.5 → gTTS (TTS) → Speaker
```

## Requirements

- Microphone and speakers/headphones
- OpenAI API key
- Whisper `base` model downloads automatically on first run (~150MB)
