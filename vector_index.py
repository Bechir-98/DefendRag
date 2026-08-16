from sentence_transformers import SentenceTransformer
import numpy as np
import json

from config import EMBEDDING_MODEL,STORAGE_DIR,EMBEDDINGS_PATH,CHUNK_IDS_PATH,CHUNKS_PATH


def build_vector_index(chunks):
    model=SentenceTransformer(EMBEDDING_MODEL)
    texts=[chunk["text"] for chunk in chunks]

    embeddings=model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings=np.array(embeddings)

    STORAGE_DIR.mkdir(parents=True,exist_ok=True)

    np.save(EMBEDDINGS_PATH,embeddings)

    np.save(
        CHUNK_IDS_PATH,
        np.array([chunk["chunk_id"] for chunk in chunks])
    )

    chunks_dict={chunk["chunk_id"]:chunk for chunk in chunks}

    with open(CHUNKS_PATH,"w") as f:
        json.dump(chunks_dict,f)

    return embeddings