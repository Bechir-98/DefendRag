from typing import TypedDict
from sentence_transformers import SentenceTransformer,CrossEncoder
from search import hybrid_search
from generate import generate
from config import TOP_K,EMBEDDING_MODEL


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
        _cross=CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    return _cross


class State(TypedDict):
    query:str
    results:list[dict]
    context:str
    answer:str
    sources:list[dict]


def retrieve(state:State):
    model=_get_model()
    results=hybrid_search(state["query"],model,top_k=TOP_K*2)
    return {"results":results}


def rerank(state:State):
    results=state["results"]

    if len(results)<=TOP_K:
        return {"results":results}

    cross=_get_cross()
    pairs=[(state["query"],r["text"]) for r in results]
    scores=cross.predict(pairs)

    for r,s in zip(results,scores):
        r["score"]=float(s)

    return {"results":sorted(results,key=lambda x:x["score"],reverse=True)}


def build_context(state:State):
    parts=[]

    for r in state["results"]:
        parts.append(f"[Source: {r['source']}, Page {r['page']}]\n{r['text']}")

    context="\n\n".join(parts)
    return {"context":context}


def generate_answer(state:State):
    answer=generate(state["query"],state["context"])
    return {"answer":answer}


def format_output(state:State):
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
    if not state["results"]:
        return "end"

    return "rerank"


from langgraph.graph import StateGraph,START,END


graph=StateGraph(State)

graph.add_node("retrieve",retrieve)
graph.add_node("rerank",rerank)
graph.add_node("build_context",build_context)
graph.add_node("generate_answer",generate_answer)
graph.add_node("format_output",format_output)

graph.add_edge(START,"retrieve")
graph.add_conditional_edges("retrieve",route_after_retrieve,{"rerank":"rerank","end":END})
graph.add_edge("rerank","build_context")
graph.add_edge("build_context","generate_answer")
graph.add_edge("generate_answer","format_output")
graph.add_edge("format_output",END)

app=graph.compile()


if __name__=="__main__":
    result=app.invoke({"query":"What is prompt injection?"})
    print("Answer:",result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  - {s['source']}, page {s['page']} (score: {s['score']:.4f})")
