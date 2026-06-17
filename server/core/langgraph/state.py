from typing import TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):

    question: str

    rewritten_question: str

    chat_history: list

    model_provider: str
    model_name: str

    documents: list[Document]

    context: str

    decision: str

    answer: str

    grounded: str

    retry_count: int