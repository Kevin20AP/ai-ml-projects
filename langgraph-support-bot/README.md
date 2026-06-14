# LangGraph Support Bot

A multi-agent customer support chatbot built with **LangGraph** — classifies user intent and routes messages to specialized handlers or human escalation.

## Features

- **Intent classification** — billing, technical, general, or unknown
- **Conditional routing** — LangGraph state machine directs each message to the right handler
- **Specialized agents** — dedicated billing, technical, and general support prompts
- **Human escalation** — low-confidence or unknown intents get escalated with a ticket ID

## Stack

LangGraph · LangChain · OpenAI GPT-3.5 · Python

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

## Run

```bash
python support_bot.py
```

Type `quit` to exit.

## Graph Flow

```
User Message → Classify Intent → Route
                                    ├─ billing   → Billing Handler  → END
                                    ├─ technical → Technical Handler → END
                                    ├─ general   → General Handler  → END
                                    └─ escalate  → Human Escalation → END
```

Escalation triggers when confidence is below 0.6 or intent is `unknown`.

## Try These

- `I was charged twice this month` → billing
- `My app keeps crashing on login` → technical
- `How do I reset my password?` → general
- `asdfghjkl` → escalation
