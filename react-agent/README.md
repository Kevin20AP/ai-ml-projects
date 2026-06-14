# ReAct Agent

A **Reasoning + Acting** agent built with LangGraph that uses tools to answer questions — calculator, weather lookup, and Wikipedia search.

## Features

- **ReAct loop** — think → act → observe → answer
- **Calculator tool** — evaluates math expressions
- **Weather tool** — live weather via wttr.in API
- **Wikipedia tool** — factual lookups with disambiguation handling
- **Transparent reasoning** — prints thought, action, and observation steps

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
python react_agent.py
```

Type `quit` to exit.

## ReAct Flow

```
Question → LLM decides tool → Tool executes → Observation → Final Answer
```

## Try These

- `What is the weather in Dallas?`
- `What is 15% of 340?`
- `Who is Elon Musk?`
- `What is the weather in Tokyo and what is 25% of 200?`
