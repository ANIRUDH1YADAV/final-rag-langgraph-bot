from langgraph.graph import StateGraph, START, END

from .state import GraphState

from .nodes import (
    retrieve_documents,
    eval_each_doc_node,
    route_after_eval,
    rewrite_query_node,
    web_search_node,
    refine,
    generate,
)

workflow = StateGraph(GraphState)

workflow.add_node(
    "retrieve",
    retrieve_documents
)

workflow.add_node(
    "eval_each_doc",
    eval_each_doc_node
)

workflow.add_node(
    "rewrite_query",
    rewrite_query_node
)

workflow.add_node(
    "web_search",
    web_search_node
)

workflow.add_node(
    "refine",
    refine
)

workflow.add_node(
    "generate",
    generate
)

workflow.add_edge(
    START,
    "retrieve"
)

workflow.add_edge(
    "retrieve",
    "eval_each_doc"
)

workflow.add_conditional_edges(
    "eval_each_doc",
    route_after_eval,
    {
        "refine": "refine",
        "rewrite_query": "rewrite_query",
    }
)

workflow.add_edge(
    "rewrite_query",
    "web_search"
)

workflow.add_edge(
    "web_search",
    "refine"
)

workflow.add_edge(
    "refine",
    "generate"
)

workflow.add_edge(
    "generate",
    END
)

graph = workflow.compile()