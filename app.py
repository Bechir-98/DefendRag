import streamlit as st
from graph import app

st.title("DefendRag")
st.caption("AI Security Knowledge Retrieval")

EXAMPLES=[
    "What is prompt injection?",
    "How does OWASP classify LLM security risks?",
    "What are the MITRE ATT&CK techniques for AI systems?",
    "How to mitigate jailbreak attacks on LLMs?",
]

cols=st.columns(len(EXAMPLES))
for i,example in enumerate(EXAMPLES):
    if cols[i].button(example,key=f"example_{i}"):
        st.session_state["query"]=example

query=st.text_input(
    "Ask a question about AI security",
    value=st.session_state.get("query",""),
    key="input"
)

if query:
    try:
        with st.spinner("Searching and generating answer..."):
            result=app.invoke({"query":query})

        if not result.get("answer"):
            st.warning("No answer generated. Try rephrasing your question.")
        else:
            st.subheader("Answer")
            st.write(result["answer"])

        if result.get("sources"):
            st.subheader("Sources")
            for i,s in enumerate(result["sources"],1):
                with st.expander(f"{i}. {s['source']} — Page {s['page']} (score: {s['score']:.4f})"):
                    st.write(s["text"])
        else:
            st.info("No sources found for this query.")

    except Exception as e:
        st.error(f"Error: {e}")
