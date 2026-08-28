import streamlit as st
from graph import app

st.title("DefendRag")
st.caption("AI Security Knowledge Retrieval")

query=st.text_input("Ask a question about AI security")

if query:
    with st.spinner("Searching and generating answer..."):
        result=app.invoke({"query":query})

    st.subheader("Answer")
    st.write(result["answer"])

    if result["sources"]:
        st.subheader("Sources")
        for i,s in enumerate(result["sources"],1):
            with st.expander(f"{i}. {s['source']} — Page {s['page']} (score: {s['score']:.4f})"):
                st.write(s["text"])
