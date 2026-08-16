from sentence_transformers import SentenceTransformer
import numpy as np

model=SentenceTransformer("all-MiniLM-L6-v2")


def build_vector_index(chunks):
    texts=[chunk["text"] for chunk in chunks]

    embeddings=model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings=np.array(embeddings)

    np.save("storage/embeddings.npy",embeddings)

    np.save(
        "storage/chunk_ids.npy",
        np.array([chunk["chunk_id"] for chunk in chunks])
    )

    return embeddings