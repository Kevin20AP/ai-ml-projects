# Multi-Agent Research System

A **LangGraph** pipeline with three collaborating agents — Researcher, Writer, and Critic — that produce polished reports with automatic revision loops.

## Features

- **Researcher agent** — gathers comprehensive facts and key concepts on a topic
- **Writer agent** — transforms research into a structured professional report
- **Critic agent** — reviews quality and approves or requests revisions
- **Revision loop** — up to 2 revision cycles before finalizing
- **Shared state** — topic, research, report, and feedback flow through the graph

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
python research_system.py
```

Type `quit` to exit.

## Agent Flow

```
Topic → Researcher → Writer → Critic
                              ├─ APPROVED → Finalize → END
                              └─ NEEDS REVISION → Revise → Writer (loop, max 2x)
```

## Report Structure

Each report includes:
- Executive Summary
- Key Findings
- Detailed Analysis
- Conclusion

## Try These Topics

- `Impact of RAG on enterprise AI`
- `How do AI agents work`
- `Future of large language models`
