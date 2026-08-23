import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]
OPENAI_API_BASE=os.environ.get("OPENAI_API_BASE","http://localhost:11434/v1")
MODEL_NAME=os.environ.get("MODEL_NAME","qwen2.5:7b")
TOP_K=int(os.environ.get("TOP_K","5"))
EMBEDDING_MODEL=os.environ.get("EMBEDDING_MODEL","all-MiniLM-L6-v2")
RERANKER_MODEL=os.environ.get("RERANKER_MODEL","cross-encoder/ms-marco-MiniLM-L6-v2")

DATA_DIR=Path(os.environ.get("DATA_DIR","data"))
STORAGE_DIR=Path(os.environ.get("STORAGE_DIR","storage"))
DB_PATH=STORAGE_DIR/"rag.db"
EMBEDDINGS_PATH=STORAGE_DIR/"embeddings.npy"
CHUNK_IDS_PATH=STORAGE_DIR/"chunk_ids.npy"
CHUNKS_PATH=STORAGE_DIR/"chunks.json"
