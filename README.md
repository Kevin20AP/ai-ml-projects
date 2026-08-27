# AI / ML Projects

End-to-end AI and machine learning projects built by **Kevin.AP** — covering LLM applications, RAG pipelines, computer vision, and intelligent systems.

## Projects

| Project | Description | Stack |
|---------|-------------|-------|
| [rag-hybrid-chatbot](./rag-hybrid-chatbot) | RAG chatbot with hybrid search (ChromaDB + BM25), multi-turn memory, and RAGAS evaluation | LangChain · OpenAI · ChromaDB · BM25 |
| [langgraph-support-bot](./langgraph-support-bot) | Multi-agent support bot with intent classification, conditional routing, and human escalation | LangGraph · LangChain · OpenAI |
| [react-agent](./react-agent) | ReAct agent with calculator, weather, and Wikipedia tools using reasoning + acting loop | LangGraph · LangChain · OpenAI |
| [huggingface-fine-tuning](./huggingface-fine-tuning) | Fine-tune GPT-2 on AI Q&A data with HuggingFace Trainer and base vs fine-tuned comparison | HuggingFace · PyTorch · GPT-2 |
| [multi-agent-research-system](./multi-agent-research-system) | Researcher, Writer, and Critic agents with revision loop for polished report generation | LangGraph · LangChain · OpenAI |
| [voice-assistant](./voice-assistant) | Hands-free voice assistant with Whisper STT, GPT-3.5, and gTTS speech output | Whisper · OpenAI · gTTS · Python |
| [sensitive-data-detection](./sensitive-data-detection) | ML system to detect sensitive data exposure in images using CNN + OCR with web deployment | TensorFlow · Flask · Tesseract OCR |
| [fake-job-prediction](./fake-job-prediction) | Classify job postings as real or fake using NLP and machine learning | Python · Scikit-learn · Jupyter |

## Architecture Overview

```
Documents → Chunking → Embeddings + Keyword Index → Hybrid Retrieval → LLM Generation
Images → CNN Classification + OCR Text Extraction → Sensitivity Detection
```

Each project follows production-ready patterns with clear pipelines, evaluation metrics, and deployable outputs.

## Author

**Kevin.AP** — Data Engineer · Data Analyst · AI/ML Engineer  
[GitHub](https://github.com/Kevin20AP) · [LinkedIn](https://www.linkedin.com/in/kevin-ap11)
