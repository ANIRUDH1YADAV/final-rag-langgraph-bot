from urllib import response

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


def rewrite_query(state: GraphState):

    llm = get_llm(
        state["model_provider"],
        state["model_name"]
    )

    prompt = f"""
You are an expert query rewriter for RAG systems.

Your job is to rewrite vague, short, or conversational
questions into specific retrieval-friendly queries.

Rules:
- Preserve the original meaning.
- Expand pronouns when possible.
- Make the query more specific.
- Use names, entities, and context from the question.
- Return only the rewritten query.

Question:
{state["question"]}
"""

    response = llm.invoke(prompt)

    rewritten_question = (
        response.content.strip()
    )

    print(
        f"\nORIGINAL QUESTION: {state['question']}"
    )

    print(
        f"\nREWRITTEN QUESTION: {rewritten_question}"
    )

    return {
        "rewritten_question": rewritten_question
    }




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
        "context": context
    }


def grade_documents(state: GraphState):

    llm = get_llm(
        state["model_provider"],
        state["model_name"]
    )

    grading_prompt = f"""
You are a relevance grader.

Question:
{state["question"]}

Retrieved Context:
{state["context"]}

Determine whether the retrieved context is relevant to answering the question.

Rules:

- If the context contains information related to the question, return YES.
- If the context contains the exact answer, return YES.
- If the context contains partial information useful for answering, return YES.
- Return NO only if the context is completely unrelated.

Return only one word:

YES

or

NO
"""

    response = llm.invoke(
        grading_prompt
    )

    print(
        "\nQUESTION:",
        state["question"]
    )

    print(
        "\nGRADE RESPONSE:",
        response.content
    )

    decision = response.content.lower()

    if "yes" in decision:
        decision = "yes"
    else:
        decision = "no"

    print(
        f"DOCUMENT GRADE DECISION: {decision}"
    )

    return {
        "decision": decision
    }


def route_documents(state: GraphState):

    print(
        f"DOCUMENT GRADE DECISION: {state['decision']}"
    )

    if state["decision"] == "yes":
        return "generate"

    return "web_search"


def web_search(state: GraphState):

    print(
        "========== TAVILY SEARCH =========="
    )

    client = TavilyClient(
        api_key=TAVILY_API_KEY
    )

    results = client.search(
        query=state["question"],
        max_results=5
    )

    web_context = "\n\n".join(
        result["content"]
        for result in results["results"]
    )

    return {
        "context": web_context
    }


def generate_answer(state: GraphState):

    llm = get_llm(
        state["model_provider"],
        state["model_name"]
    )

    prompt = get_prompt()

    print(
    "\nCHAT HISTORY PASSED TO LLM:\n",
    state.get("chat_history", [])
)

    prompt_value = prompt.invoke(
        {
            "context": state["context"],
            "chat_history": state.get(
                "chat_history",
                []
            ),
            "input": state["question"]
        }
    )

    response = llm.invoke(
        prompt_value
    )

    return {
        "answer": response.content,
        "retry_count": state.get(
            "retry_count",
            0
        ) + 1
    }

def hallucination_check(state: GraphState):

    llm = get_llm(
        state["model_provider"],
        state["model_name"]
    )

    prompt = f"""
You are a strict hallucination detector.

Question:
{state["question"]}

Context:
{state["context"]}

Answer:
{state["answer"]}

Rules:

- Return "yes" if every important claim in the answer is supported by the context.
- Return "no" if the answer contains unsupported information.
- Return only one word: yes or no.
"""

    response = llm.invoke(prompt)

    grounded = (
        response.content
        .strip()
        .lower()
    )

    print(
        f"HALLUCINATION CHECK: {grounded}"
    )

    return {
        "grounded": grounded
    }

def update_memory(state: GraphState):

    history = state.get(
        "chat_history",
        []
    )

    history.append(
        {
            "role": "user",
            "content": state["question"]
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": state["answer"]
        }
    )

    return {
        "chat_history": history
    }




def route_hallucination(state: GraphState):

    retry_count = state.get(
        "retry_count",
        0
    )

    if state["grounded"] == "yes":
        return "end"

    if retry_count >= 2:
        return "end"

    return "regenerate"