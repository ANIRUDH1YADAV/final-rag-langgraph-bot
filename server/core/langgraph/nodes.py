from urllib import response
from langchain_core.documents import Document

from click import Context
from streamlit import context

from .state import GraphState

from core.vector_database import load_vectorstore
from core.llm_chain_factory import (
    get_llm,
    get_prompt
)

from tavily import TavilyClient
from config.settings import TAVILY_API_KEY

import re

from typing import List
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

from typing import List
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

from config.settings import UPPER_TH, LOWER_TH






from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults




from langchain_core.prompts import ChatPromptTemplate







def retrieve_documents(state: GraphState):

    question = state.get(
    "rewritten_question",
    state["question"]
)

    model_provider = state["model_provider"]

    vectorstore = load_vectorstore(
        model_provider
    )

    retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5
    }
)

    docs = retriever.invoke(
        question
    )

    print("\n========== RETRIEVED DOCS ==========\n")

    for index, doc in enumerate(docs):

        print(f"\n--- DOCUMENT {index + 1} ---\n")

        print(doc.page_content[:500])

        print("\n=============================\n")

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return {
    "documents": docs,
    "context": context,
    "source": "document"
}


class DocEvalScore(BaseModel):
    score: float
    reason: str


def eval_each_doc_node(state: GraphState):

    llm = get_llm(
        state["model_provider"],
        state["model_name"]
    )

    doc_eval_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict retrieval evaluator for RAG.\n"
                "You will be given ONE retrieved chunk and a question.\n"
                "Return a relevance score in [0.0, 1.0].\n"
                "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
                "- 0.0: chunk is irrelevant\n"
                "Be conservative with high scores.\n"
                "Also return a short reason.\n"
                "Output JSON only."
            ),
            (
                "human",
                "Question: {question}\n\nChunk:\n{chunk}"
            ),
        ]
    )

    doc_eval_chain = (
        doc_eval_prompt
        | llm.with_structured_output(DocEvalScore)
    )

    question = state.get(
        "rewritten_question",
        state["question"]
    )

    scores = []
    good_docs = []

    for doc in state["documents"]:

        result = doc_eval_chain.invoke(
            {
                "question": question,
                "chunk": doc.page_content
            }
        )

        scores.append(result.score)

        print(
            f"Chunk Score: {result.score:.2f}"
        )

        if result.score > LOWER_TH:
            good_docs.append(doc)

    if not scores:
        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": "No documents retrieved."
        }

    if any(score > UPPER_TH for score in scores):

        print(
            "RETRIEVAL DECISION: CORRECT"
        )

        return {
            "good_docs": good_docs,
            "verdict": "CORRECT",
            "reason": (
                f"At least one retrieved chunk "
                f"scored > {UPPER_TH}."
            )
        }

    if all(score < LOWER_TH for score in scores):

        print(
            "RETRIEVAL DECISION: INCORRECT"
        )

        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": (
                f"All retrieved chunks "
                f"scored < {LOWER_TH}."
            )
        }

    print(
        "RETRIEVAL DECISION: AMBIGUOUS"
    )

    return {
        "good_docs": good_docs,
        "verdict": "AMBIGUOUS",
        "reason": (
            f"No chunk scored > {UPPER_TH}, "
            f"but not all were < {LOWER_TH}."
        )
    }


def route_after_eval(state: GraphState):

    if state["verdict"] == "CORRECT":
        return "refine"

    return "rewrite_query"


class WebQuery(BaseModel):
    query: str


def rewrite_query_node(state: GraphState):

    llm = get_llm(
        state["model_provider"],
        state["model_name"]
    )

    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rewrite the user question into a web search query composed of keywords.\n"
                "Rules:\n"
                "- Keep it short (6–14 words).\n"
                "- If the question implies recency (e.g., recent/latest/last week/last month), "
                "add a constraint like (last 30 days).\n"
                "- Do NOT answer the question.\n"
                "- Return JSON with a single key: query"
            ),
            (
                "human",
                "Question: {question}"
            ),
        ]
    )

    rewrite_chain = (
        rewrite_prompt
        | llm.with_structured_output(WebQuery)
    )

    result = rewrite_chain.invoke(
        {
            "question": state["question"]
        }
    )

    print(
        f"\nORIGINAL QUESTION: {state['question']}"
    )

    print(
        f"\nWEB QUERY: {result.query}"
    )

    return {
        "web_query": result.query
    }


tavily = TavilySearchResults(max_results=5)


def web_search_node(state: GraphState):

    query = (
        state.get("web_query")
        or state.get("rewritten_question")
        or state["question"]
    )

    results = tavily.invoke(
        {"query": query}
    )

    web_docs = []

    for result in results or []:

        title = result.get(
            "title",
            ""
        )

        url = result.get(
            "url",
            ""
        )

        content = (
            result.get("content", "")
            or result.get("snippet", "")
        )

        text = (
            f"TITLE: {title}\n"
            f"URL: {url}\n"
            f"CONTENT:\n{content}"
        )

        web_docs.append(
            Document(
                page_content=text,
                metadata={
                    "url": url,
                    "title": title
                }
            )
        )

    return {
        "web_docs": web_docs
    }

class KeepOrDrop(BaseModel):
    keep: bool

def decompose_to_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [
        s.strip()
        for s in sentences
        if len(s.strip()) > 20
    ]


def refine(state: GraphState):

    llm = get_llm(
        state["model_provider"],
        state["model_name"]
    )

    filter_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict relevance filter.\n"
                "Return keep=true only if the sentence directly helps answer the question.\n"
                "Use ONLY the sentence. Output JSON only."
            ),
            (
                "human",
                "Question: {question}\n\nSentence:\n{sentence}"
            ),
        ]
    )

    filter_chain = (
        filter_prompt
        | llm.with_structured_output(KeepOrDrop)
    )

    question = state["question"]

    if state.get("verdict") == "CORRECT":

        docs_to_use = state["good_docs"]

    elif state.get("verdict") == "INCORRECT":

        docs_to_use = state["web_docs"]

    else:  # AMBIGUOUS

        docs_to_use = (
            state["good_docs"]
            + state["web_docs"]
        )

    context = "\n\n".join(
        doc.page_content
        for doc in docs_to_use
    ).strip()

    strips = decompose_to_sentences(
        context
    )

    kept: List[str] = []
    for s in strips:
        if filter_chain.invoke({"question": question, "sentence": s}).keep:
            kept.append(s)

    

    refined_context = "\n".join(
        kept
    ).strip()

    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_context": refined_context,
    }






def generate(state: GraphState):

    llm = get_llm(
        state["model_provider"],
        state["model_name"]
    )

    answer_prompt = ChatPromptTemplate.from_messages(
        [
           (
    "system",
    """
You are a Contextual Retrieval-Augmented Generation (CRAG) assistant.

Rules:

1. Answer ONLY using the provided context.

2. Never use your own knowledge.

3. If the context does not contain enough information to answer the question, respond exactly:
"I don't know based on the provided context."

4. If the user asks whether the uploaded document, PDF, or book mentions, discusses, explains, or contains a topic, answer ONLY based on the document context.

5. Never assume that information found through web search exists in the uploaded document.

6. If the uploaded document does not discuss the requested topic, respond exactly:
"The uploaded document does not discuss this topic."

7. Do not fabricate, infer, or hallucinate information.

8. If the context comes from web search, answer only using the retrieved web context.

9. If the context comes from the uploaded document, answer only using the uploaded document context.
"""
),
            (
                "human",
                "Question: {question}\n\nContext:\n{context}"
            ),
        ]
    )

    chain = answer_prompt | llm

    response = chain.invoke(
        {
            "question": state["question"],
            "context": state["refined_context"]
        }
    )

    return {
        "answer": response.content
    }   







