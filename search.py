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