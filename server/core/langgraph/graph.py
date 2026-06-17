from langgraph.graph import StateGraph, END

from .state import GraphState

from .nodes import (
    rewrite_query,
    retrieve_documents,
    grade_documents,
    route_documents,
    update_memory,
    web_search,
    generate_answer,
    hallucination_check,
    route_hallucination
)
workflow = StateGraph(GraphState)

workflow.add_node(
    "rewrite",
    rewrite_query
)

workflow.add_node(
    "retrieve",
    retrieve_documents
)

workflow.add_node(
    "grade",
    grade_documents
)

workflow.add_node(
    "web_search",
    web_search
)

workflow.add_node(
    "generate",
    generate_answer
)


workflow.add_node(
    "memory",
    update_memory
)


workflow.add_node(
    "hallucination_check",
    hallucination_check
)

workflow.set_entry_point(
    "rewrite"
)

workflow.add_edge(
    "rewrite",
    "retrieve"
)

workflow.add_edge(
    "retrieve",
    "grade"
)

workflow.add_conditional_edges(
    "grade",
    route_documents,
    {
        "generate": "generate",
        "web_search": "web_search"
    }
)

workflow.add_edge(
    "web_search",
    "generate"
)

workflow.add_edge(
    "generate",
    "hallucination_check"
)


workflow.add_edge(
    "generate",
    "memory"
)

workflow.add_node(
    "route_hallucination",
    route_hallucination
)

workflow.add_conditional_edges(
    "hallucination_check",
    route_hallucination,
    {
        "end": END,
        "regenerate": "generate"
    }
)

graph = workflow.compile()