import sqlite3
from pathlib import Path

DB_PATH=Path("storage/rag.db")


def create_keyword_index():
    DB_PATH.parent.mkdir(parents=True,exist_ok=True)

    conn=sqlite3.connect(DB_PATH)

    conn.execute("DROP TABLE IF EXISTS chunks_fts")

    conn.execute("""
    CREATE VIRTUAL TABLE chunks_fts
    USING fts5(
        text,
        source UNINDEXED,
        path UNINDEXED,
        page UNINDEXED,
        chunk_id UNINDEXED
    )
""")

    conn.commit()

    return conn


def index_chunks(conn,chunks):
    conn.executemany(
        """
        INSERT INTO chunks_fts
        (text,source,path,page,chunk_id)
        VALUES (?,?,?,?,?)
        """,
        [
            (
                chunk["text"],
                chunk["source"],
                chunk["path"],
                chunk["page"],
                chunk["chunk_id"]
            )
            for chunk in chunks
        ]
    )

    conn.commit()