"""
Day 1 — Your first streaming LLM app.
Goal: send a prompt to Claude and watch the reply stream in token-by-token.

SETUP (one time):
  1. Get an API key: https://console.anthropic.com  ->  Settings -> API Keys
  2. In your terminal:
       pip install anthropic
       export ANTHROPIC_API_KEY="sk-ant-...yourkey..."     # Mac/Linux
  3. Run it:
       python streaming_chat.py

Then try changing the prompt, or the system message, and re-run.
This is the seed of Project 1 (AI Chat App with Tools).
"""

from dotenv import load_dotenv
load_dotenv()
import os
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def stream_reply(user_message: str, system: str = "You are a concise, helpful assistant."):
    """Send one message and stream the response to the terminal."""
    print("\nClaude: ", end="", flush=True)
    with client.messages.stream(
        model="claude-sonnet-4-5",          # a fast, capable model
        max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:      # <-- this is the streaming magic
            print(text, end="", flush=True)
    print("\n")


def main():
    print("=== Streaming chat (type 'quit' to exit) ===")
    while True:
        try:
            msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg.lower() in {"quit", "exit"}:
            break
        if msg:
            stream_reply(msg)
    print("Done. Nice work shipping Day 1. Commit this and push to GitHub!")


if __name__ == "__main__":
    main()