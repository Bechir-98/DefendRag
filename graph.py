from typing import TypedDict
from sentence_transformers import SentenceTransformer,CrossEncoder
from langgraph.graph import StateGraph,START,END
from search import hybrid_search
from generate import generate,judge_evidence,rewrite_query,grade_answer
from config import TOP_K,MAX_RETRIES,MAX_GRADE_RETRIES,EMBEDDING_MODEL,RERANKER_MODEL


# ponytail: load model once at module level, reload if embedding model changes
_model=None
# ponytail: cross-encoder downloaded once by sentence-transformers on first use; swap for a domain-tuned reranker if rankings plateau
_cross=None

def _get_model():
    global _model
    if _model is None:
        _model=SentenceTransformer(EMBEDDING_MODEL)
    return _model

def _get_cross():
    global _cross
    if _cross is None:
        _cross=CrossEncoder(RERANKER_MODEL)
    return _cross


class State(TypedDict):
    query:str
    search_query:str
    retries:int
    sufficient:bool
    critique:str
    grade_retries:int
    grade_passed:bool
    results:list[dict]
    context:str
    answer:str
    sources:list[dict]


def _evidence_block(results,limit=3,chars=200):
    return "\n\n".join(
        f"[Source: {r['source']}, Page {r['page']}]\n{r['text'][:chars]}"
        for r in results[:limit]
    )


def retrieve(state:State):
    """Fetch TOP_K*2 hybrid candidates; retries accumulate, deduped by chunk_id."""
    model=_get_model()
    q=state.get("search_query") or state["query"]
    fresh=hybrid_search(q,model,top_k=TOP_K*2)
    seen={r["chunk_id"] for r in state.get("results") or []}
    merged=list(state.get("results") or [])
    merged.extend(r for r in fresh if r["chunk_id"] not in seen)
    return {"results":merged}


def rerank(state:State):
    """Cross-encoder rescore; keep only the top-K for generation."""
    results=state["results"]

    if len(results)<=TOP_K:
        return {"results":results}

    cross=_get_cross()
    pairs=[(state["query"],r["text"]) for r in results]
    scores=cross.predict(pairs)

    for r,s in zip(results,scores):
        r["score"]=float(s)

    return {"results":sorted(results,key=lambda x:x["score"],reverse=True)[:TOP_K]}


def build_context(state:State):
    """Format ranked chunks as [Source, Page] blocks for the LLM."""
    parts=[]

    for r in state["results"][:TOP_K]:
        parts.append(f"[Source: {r['source']}, Page {r['page']}]\n{r['text']}")

    context="\n\n".join(parts)
    return {"context":context}


def check_evidence(state:State):
    """Ask the LLM whether the top-K passages suffice; fail open to generation."""
    evidence=_evidence_block(state["results"],limit=TOP_K,chars=500)
    return {"sufficient":judge_evidence(state["query"],evidence)}


def rewrite(state:State):
    """Reformulate the search query; counts a retry even on LLM failure."""
    retries=(state.get("retries") or 0)+1
    evidence=_evidence_block(state["results"])
    q=rewrite_query(state["query"],evidence)

    if not q or q.startswith("LLM error:"):
        q=state.get("search_query") or state["query"]

    return {"search_query":q.strip().strip('"') or state["query"],"retries":retries}


def generate_answer(state:State):
    """Answer from context only; context is untrusted (see generate.py)."""
    critique=state.get("critique")
    answer=generate(state["query"],state["context"],critique=critique)
    out={"answer":answer}

    if critique:
        out["grade_retries"]=(state.get("grade_retries") or 0)+1

    return out


def grade(state:State):
    """LLM review of the answer; fail open to format_output."""
    passed,critique=grade_answer(state["query"],state["context"],state["answer"])
    return {"grade_passed":passed,"critique":critique}


def format_output(state:State):
    """Trim chunks to short source cards for the UI."""
    sources=[]

    for r in state["results"]:
        sources.append({
            "source":r["source"],
            "page":r["page"],
            "score":r["score"],
            "text":r["text"][:200]
        })

    return {"sources":sources}


def route_after_retrieve(state:State):
    """Skip rerank/generate when retrieval found nothing."""
    if not state["results"]:
        return "end"

    return "rerank"


def route_after_check(state:State):
    """Loop back for a rewrite when evidence is thin, else generate."""
    if not state.get("sufficient",True) and (state.get("retries") or 0)<MAX_RETRIES:
        return "rewrite_query"

    return "build_context"


def route_after_grade(state:State):
    """Loop back for one revision on FAIL, else ship the answer."""
    if not state.get("grade_passed",True) and (state.get("grade_retries") or 0)<MAX_GRADE_RETRIES:
        return "generate_answer"

    return "format_output"


graph=StateGraph(State)

graph.add_node("retrieve",retrieve)
graph.add_node("rerank",rerank)
graph.add_node("check_evidence",check_evidence)
graph.add_node("rewrite",rewrite)
graph.add_node("build_context",build_context)
graph.add_node("generate_answer",generate_answer)
graph.add_node("grade",grade)
graph.add_node("format_output",format_output)

graph.add_edge(START,"retrieve")
graph.add_conditional_edges("retrieve",route_after_retrieve,{"rerank":"rerank","end":END})
graph.add_edge("rerank","check_evidence")
graph.add_conditional_edges("check_evidence",route_after_check,{"rewrite_query":"rewrite","build_context":"build_context"})
graph.add_edge("rewrite","retrieve")
graph.add_edge("build_context","generate_answer")
graph.add_edge("generate_answer","grade")
graph.add_conditional_edges("grade",route_after_grade,{"generate_answer":"generate_answer","format_output":"format_output"})
graph.add_edge("format_output",END)

app=graph.compile()


if __name__=="__main__":
    result=app.invoke({"query":"What is prompt injection?"})
    print("Answer:",result["answer"])
    print(f"Retries: {result.get('retries',0)} (retrieval) / {result.get('grade_retries',0)} (grade)")
    print("\nSources:")
    for s in result["sources"]:
        print(f"  - {s['source']}, page {s['page']} (score: {s['score']:.4f})")
