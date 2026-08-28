# DefendRag

Agentic RAG system for AI security knowledge retrieval. Combines hybrid search (keyword + vector) with LLM generation to answer cybersecurity questions from trusted sources.

## Setup

```bash
cp .env.example .env
# Edit .env with your OpenAI-compatible API key and endpoint
```

## Usage

```bash
# 1. Index your documents (place PDFs in data/)
python index.py

# 2. Run the UI
streamlit run app.py
```

## Architecture

```
Query → LangGraph Agent → Hybrid Search → Rerank → LLM → Answer + Sources
```

- **Hybrid Search**: SQLite FTS5 (keyword) + SentenceTransformer embeddings (vector)
- **Agent**: LangGraph with retrieve → rerank → generate pipeline
- **LLM**: Any OpenAI-compatible API (Ollama, vLLM, OpenAI, etc.)
