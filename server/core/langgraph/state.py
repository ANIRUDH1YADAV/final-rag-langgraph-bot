from typing import TypedDict, List
from langchain_core.documents import Document


class GraphState(TypedDict, total=False):

    # User Input
    question: str

    # Model Settings
    model_provider: str
    model_name: str

    # Retrieval
    documents: List[Document]
    context: str

    # Evaluation
    good_docs: List[Document]
    verdict: str
    reason: str

    # Web Search
    web_query: str
    web_docs: List[Document]

    # Refinement
    strips: List[str]
    kept_strips: List[str]
    refined_context: str

    # Final Output
    answer: str