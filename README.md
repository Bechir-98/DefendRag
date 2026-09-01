# DefendRag

Agentic RAG system for AI security knowledge retrieval. Combines hybrid search (keyword + vector), a cross-encoder reranker, and LLM generation to answer cybersecurity questions from trusted sources — with a built-in evaluation harness.

## Setup

```bash
cp .env.example .env
# Edit .env with your OpenAI-compatible API key and endpoint
```

## Usage

```bash
# 1. Index your documents (place PDFs in data/ or any subdirectory)
#    Supported structure:
#    data/owasp/    file1.pdf
#    data/mitre/    file2.pdf
#    data/nist/     file3.pdf
python index.py

# 2. Run the UI
streamlit run app.py

# 3. Evaluate retrieval + answer quality
python eval.py
```

The first run auto-downloads the embedding and cross-encoder models to `~/.cache/huggingface`.

## Architecture

```
Query → LangGraph Agent → Hybrid Search → Cross-Encoder Rerank → LLM → Answer + Sources
```

- **Hybrid Search**: SQLite FTS5 (keyword) + SentenceTransformer embeddings (vector)
- **Agent**: LangGraph with retrieve → rerank → generate pipeline
- **Rerank**: CrossEncoder (`cross-encoder/ms-marco-MiniLM-L6-v2`) rescores hybrid results; only the top-K pass to generation
- **LLM**: Any OpenAI-compatible API (Ollama, vLLM, OpenAI, etc.)
- **Eval**: `eval.py` runs a golden question set, checks doc recall, key terms in the answer, and source citation; exits non-zero on failure

## Troubleshooting

- Groq's free tier rate-limits bursts of requests (HTTP 429). `generate.py` retries with backoff; if it persists, wait ~30s and retry.
