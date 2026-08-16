import sqlite3
from keyword_index import DB_PATH


def search_keyword(query,top_k=5):
    conn=sqlite3.connect(DB_PATH)

    cursor=conn.execute(
        """
        SELECT
            chunk_id,
            text,
            source,
            path,
            page,
            bm25(chunks_fts) AS score
        FROM chunks_fts
        WHERE chunks_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (query,top_k)
    )

    results=cursor.fetchall()

    conn.close()

    return results

def search_vector(query,embeddings,chunk_ids,chunks,top_k=5):
    query_embedding=model.encode(
        query,
        normalize_embeddings=True
    )

    scores=embeddings @ query_embedding

    top_indices=np.argsort(scores)[::-1][:top_k]

    results=[]

    for i in top_indices:
        chunk=chunks[chunk_ids[i]]

        results.append({
            "chunk_id":chunk["chunk_id"],
            "text":chunk["text"],
            "source":chunk["source"],
            "path":chunk["path"],
            "page":chunk["page"],
            "score":float(scores[i])
        })

    return results