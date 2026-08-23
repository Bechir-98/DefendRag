import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]
OPENAI_API_BASE=os.environ.get("OPENAI_API_BASE","http://localhost:11434/v1")
MODEL_NAME=os.environ.get("MODEL_NAME","qwen2.5:7b")
TOP_K=int(os.environ.get("TOP_K","5"))
EMBEDDING_MODEL=os.environ.get("EMBEDDING_MODEL","all-MiniLM-L6-v2")
