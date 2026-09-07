import time
import json
import streamlit as st
from pathlib import Path
from graph import app
from config import MODEL_NAME,TOP_K,EMBEDDING_MODEL

st.title("DefendRag")
st.caption("AI Security Knowledge Retrieval")

EXAMPLES=[
    "What is prompt injection?",
    "How does OWASP classify LLM security risks?",
    "What are the MITRE ATT&CK techniques for AI systems?",
    "How to mitigate jailbreak attacks on LLMs?",
]

if "query" not in st.session_state:
    st.session_state["query"]=""
if "history" not in st.session_state:
    st.session_state["history"]=[]

st.sidebar.header("Configuration")
st.sidebar.write(f"**Model:** {MODEL_NAME}")
st.sidebar.write(f"**Top-K:** {TOP_K}")
st.sidebar.write(f"**Embeddings:** {EMBEDDING_MODEL}")

abs_path=Path("storage/chunks.json")
chunks=len(json.load(abs_path.open())) if abs_path.exists() else 0
pdfs=list(Path("data").rglob("*.pdf"))
st.sidebar.markdown("---")
st.sidebar.subheader("Corpus")
st.sidebar.write(f"**PDFs:** {len(pdfs)}")
st.sidebar.write(f"**Chunks:** {chunks}")

ask_tab,eval_tab=st.tabs(["Ask","Evaluation"])

with ask_tab:
    cols=st.columns(len(EXAMPLES))
    for i,example in enumerate(EXAMPLES):
        if cols[i].button(example,key=f"example_{i}"):
            st.session_state["query"]=example

    query=st.text_input(
        "Ask a question about AI security",
        value=st.session_state.get("query",""),
        key="query"
    )

    if st.button("New Chat"):
        st.session_state["history"]=[]

    if query:
        start=time.time()
        try:
            with st.spinner("Searching and generating answer..."):
                result=app.invoke({"query":query})
            latency=time.time()-start

            st.session_state["history"].append({"query":query,"result":result})

            if not result.get("answer"):
                st.warning("No answer generated. Try rephrasing your question.")
            else:
                st.subheader("Answer")
                st.code(result["answer"],language=None)
                st.caption(f"Latency: {latency:.1f}s · Sources: {len(result.get('sources',[]))} · Top: {result['sources'][0]['source'] if result.get('sources') else '—'}")

            if result.get("sources"):
                st.subheader("Sources")
                for i,s in enumerate(result["sources"],1):
                    with st.expander(f"{i}. {s['source']} — Page {s['page']} (score: {s['score']:.4f})"):
                        st.write(s["text"])
            else:
                st.info("No sources found for this query.")

        except Exception as e:
            st.error(f"Error: {e}")

    if st.session_state["history"]:
        st.subheader("History")
        with st.expander(f"{len(st.session_state['history'])} previous queries"):
            for h in st.session_state["history"][:-1]:
                st.write(f"**Q:** {h['query']}")
                st.write(f"**A:** {h['result'].get('answer') or '(empty)'}")
                st.markdown("---")

with eval_tab:
    st.caption("Runs the 6 golden cases against the live pipeline (Groq + reranker), throttled ~3s/case for rate limits.")
    if st.button("Run Evaluation"):
        from eval import run_eval
        with st.spinner("Running evaluation..."):
            results,summary=run_eval()

        col1,col2,col3=st.columns(3)
        col1.metric("Fully Pass",f"{summary['pass']}/{summary['total']}")
        col2.metric("Recall",f"{summary['recall']}/{summary['total']}")
        col3.metric("Citation",f"{summary['cite']}/{summary['total']}")

        rows=[]
        for r in results:
            rows.append({
                "query":r["query"],
                "status":r["status"],
                "recall":r["recall"],
                "term":r["term"],
                "cite":r["cite"],
                "n_sources":r["n_sources"],
                "expected_source":r["expected_source"],
            })
        st.dataframe(rows,use_container_width=True)
