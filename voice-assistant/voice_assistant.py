import os
import time
import tempfile
import sounddevice as sd
import soundfile as sf
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from gtts import gTTS
import pygame
import whisper

load_dotenv()
client = OpenAI()

# ── 1. LOAD WHISPER MODEL ─────────────────────────────────────────────────────
print("="*55)
print("🎙️  Voice AI Assistant")
print("="*55)
print("\nLoading Whisper speech recognition model...")
whisper_model = whisper.load_model("base")
print("✅ Whisper loaded")

# ── 2. INITIALIZE PYGAME FOR AUDIO PLAYBACK ───────────────────────────────────
pygame.mixer.init()
print("✅ Audio system ready")

# ── 3. CONVERSATION HISTORY ───────────────────────────────────────────────────
conversation_history = [
    {
        "role": "system",
        "content": "You are a helpful voice AI assistant. Keep responses concise and conversational — 2-3 sentences maximum since your responses will be spoken aloud."
    }
]

# ── 4. RECORD AUDIO FROM MICROPHONE ──────────────────────────────────────────
def record_audio(duration=5, sample_rate=16000):
    print(f"\n🎙️  Recording for {duration} seconds — speak now!")
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32
    )
    sd.wait()
    print("✅ Recording complete")
    return audio_data, sample_rate

# ── 5. TRANSCRIBE AUDIO WITH WHISPER ─────────────────────────────────────────
def transcribe(audio_data, sample_rate):
    print("🔄 Transcribing with Whisper...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_data, sample_rate)
        tmp_path = tmp.name

    result = whisper_model.transcribe(tmp_path)
    os.unlink(tmp_path)
    text = result["text"].strip()
    print(f"📝 You said: {text}")
    return text

# ── 6. GET GPT RESPONSE ───────────────────────────────────────────────────────
def get_response(user_text):
    print("🤖 GPT thinking...")
    conversation_history.append({
        "role": "user",
        "content": user_text
    })

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation_history,
        temperature=0.7
    )

    answer = response.choices[0].message.content
    conversation_history.append({
        "role": "assistant",
        "content": answer
    })

    print(f"💬 GPT: {answer}")
    return answer

# ── 7. SPEAK THE RESPONSE ─────────────────────────────────────────────────────
def speak(text):
    print("🔊 Speaking response...")
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(tmp_path)

    pygame.mixer.music.load(tmp_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    pygame.mixer.music.unload()
    os.unlink(tmp_path)
    print("✅ Done speaking")

# ── 8. MAIN LOOP ──────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("Ready! Press Enter to speak, type 'quit' to exit.")
print("="*55)

while True:
    user_choice = input("\nPress Enter to speak (or type 'quit'): ").strip()

    if user_choice.lower() == "quit":
        print("Goodbye!")
        break

    try:
        # Step 1 — Record
        audio, sr = record_audio(duration=5)

        # Step 2 — Transcribe
        user_text = transcribe(audio, sr)

        if not user_text:
            print("No speech detected — try again")
            continue

        # Step 3 — Get GPT response
        response = get_response(user_text)

        # Step 4 — Speak it
        speak(response)

    except Exception as e:
        print(f"Error: {str(e)}")
