from ingestion import load_all_pdfs,build_chunks
from keyword_index import create_keyword_index,index_chunks
from vector_index import build_vector_index


docs=load_all_pdfs("data")
print("Pages loaded:",len(docs))

chunks=build_chunks(docs)
print("Chunks:",len(chunks))

conn=create_keyword_index()
index_chunks(conn,chunks)
conn.close()

build_vector_index(chunks)

print("Index built successfully.")