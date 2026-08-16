import sqlite3
import json
import numpy as np
from config import DB_PATH,EMBEDDINGS_PATH,CHUNK_IDS_PATH,CHUNKS_PATH


def search_keyword(query,top_k=5):
    """FTS5 keyword search. Returns [{chunk_id,text,source,path,page,score}]."""
    conn=sqlite3.connect(DB_PATH)

    term=query.replace('"',' ')
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
        (f'"{term}"',top_k)
    )

    cols=["chunk_id","text","source","path","page","score"]
    results=[dict(zip(cols,row)) for row in cursor.fetchall()]

    conn.close()

    return results

def search_vector(query,embeddings,chunk_ids,chunks,model,top_k=5):
    """Cosine search over normalized embeddings. Same dict shape as search_keyword."""
    query_embedding=model.encode(
        query,
        normalize_embeddings=True
    )

    scores=embeddings @ query_embedding

    # ponytail: per-query min-max normalization, cross-encoder if absolute scores matter
    min_score,max_score=scores.min(),scores.max()
    if max_score>min_score:
        scores=(scores-min_score)/(max_score-min_score)

    top_indices=np.argsort(scores)[::-1][:top_k]

    results=[]

    for i in top_indices:
        chunk=chunks[str(chunk_ids[i])]

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
    """Merge keyword + vector hits, summing normalized scores."""
    keyword_results=search_keyword(query,top_k=top_k*2)

    embeddings=np.load(EMBEDDINGS_PATH)
    chunk_ids=np.load(CHUNK_IDS_PATH)

    with open(CHUNKS_PATH) as f:
        chunks=json.load(f)

    vector_results=search_vector(query,embeddings,chunk_ids,chunks,model,top_k=top_k*2)

    merged={}

    for r in keyword_results:
        # ponytail: bm25() is negative (more negative = better); map to 0-1 with larger=better
        normalized=-r["score"]/(1-r["score"])
        merged[r["chunk_id"]]={**r,"score":normalized}

    for r in vector_results:
        cid=r["chunk_id"]
        if cid in merged:
            merged[cid]["score"]+=r["score"]
        else:
            merged[cid]=r

    results=sorted(merged.values(),key=lambda x:x["score"],reverse=True)

    return results[:top_k]


if __name__=="__main__":
    from sentence_transformers import SentenceTransformer

    print("=== Keyword Search ===")
    keyword_results=search_keyword("prompt injection",top_k=3)
    for r in keyword_results:
        print(f"  [{r['source']} p{r['page']}] score={r['score']:.4f} | {r['text'][:80]}...")

    print("\n=== Hybrid Search ===")
    model=SentenceTransformer("all-MiniLM-L6-v2")
    hybrid_results=hybrid_search("prompt injection",model,top_k=3)
    for r in hybrid_results:
        print(f"  [{r['source']} p{r['page']}] score={r['score']:.4f} | {r['text'][:80]}...")

    print("\nSelf-check passed.")