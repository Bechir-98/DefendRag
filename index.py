from ingestion import load_all_pdfs,build_chunks
from keyword_index import create_keyword_index,index_chunks


docs=load_all_pdfs("data")
print("Pages loaded:",len(docs))

chunks=build_chunks(docs)
print("Chunks:",len(chunks))

conn=create_keyword_index()
index_chunks(conn,chunks)

conn.close()

print("Index built successfully.")