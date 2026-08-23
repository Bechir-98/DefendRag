import sqlite3
import json
import numpy as np
from sentence_transformers import SentenceTransformer
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

def search_vector(query,embeddings,chunk_ids,chunks,model,top_k=5):
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


def hybrid_search(query,model,top_k=5):
    keyword_results=search_keyword(query,top_k=top_k*2)

    embeddings=np.load("storage/embeddings.npy")
    chunk_ids=np.load("storage/chunk_ids.npy")

    with open("storage/chunks.json") as f:
        chunks=json.load(f)

    vector_results=search_vector(query,embeddings,chunk_ids,chunks,model,top_k=top_k*2)

    merged={}

    for chunk_id,text,source,path,page,score in keyword_results:
        normalized=1/(1+abs(score))
        merged[chunk_id]={
            "chunk_id":chunk_id,
            "text":text,
            "source":source,
            "path":path,
            "page":page,
            "score":normalized
        }

    for r in vector_results:
        cid=r["chunk_id"]
        if cid in merged:
            merged[cid]["score"]+=r["score"]
        else:
            merged[cid]=r

    results=sorted(merged.values(),key=lambda x:x["score"],reverse=True)

    return results[:top_k]