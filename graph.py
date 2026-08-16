from typing import TypedDict
from search import search_keyword
from langgraph.graph import StateGraph,START,END


class State(TypedDict):
    query:str
    results:list


def search(state:State):
    results=search_keyword(state["query"],top_k=5)

    return {"results":results}


def check_results(state:State):
    if state["results"]:
        return "found"

    return "not_found"


graph=StateGraph(State)

graph.add_node("search",search)
graph.add_node("check_results",check_results)

graph.add_edge(START,"search")
graph.add_edge("search","check_results")

graph.add_conditional_edges("check_results",check_results,{"found":END,"not_found":END})

app=graph.compile()

result=app.invoke({"query":"prompt"})

print(result)