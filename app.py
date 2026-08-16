import streamlit as st
from search import search_keyword

st.title("DefendRag")

query=st.text_input("Search your documents")

if query:
    results=search_keyword(query,top_k=5)

    for chunk_id,text,source,path,page,score in results:
        st.subheader(f"{source} — Page {page}")
        st.caption(f"Chunk: {chunk_id} | Score: {score:.4f}")
        st.write(text)