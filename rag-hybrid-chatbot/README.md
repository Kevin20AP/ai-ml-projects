# RAG Hybrid Chatbot

A Retrieval Augmented Generation (RAG) chatbot that combines **vector similarity search** (ChromaDB) with **BM25 keyword search** for improved retrieval, plus multi-turn conversation memory.

## Features

- **ChromaDB vector store** — semantic search over document embeddings
- **BM25 keyword search** — term-frequency based retrieval
- **Hybrid search** — merges vector + keyword results for better recall
- **Multi-turn memory** — maintains chat history across the session
- **RAGAS metrics** — evaluation framework reference (faithfulness, relevance, precision, recall)

## Stack

LangChain · OpenAI · ChromaDB · BM25 · Python

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

## Run

```bash
python rag_chatbot.py
```

Type `quit` to exit the chat loop.

## How It Works

```
User Query → Hybrid Search (Vector + BM25) → Context Injection → GPT-3.5 → Response
```

1. Documents are split into chunks with `RecursiveCharacterTextSplitter`
2. Embeddings are stored in ChromaDB for similarity search
3. BM25 indexes the same chunks for keyword matching
4. Hybrid search deduplicates and merges top results from both retrievers
5. Retrieved context is injected into the LLM prompt with conversation history
