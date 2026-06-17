from config.settings import GROQ_API_KEY, GOOGLE_API_KEY

from langchain_core.prompts import ChatPromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from utils.logger import logger


def get_prompt():
    return ChatPromptTemplate.from_template(
        """
You are a helpful AI assistant.

Use the conversation history whenever it contains
information needed to answer the user's question.

Priority:

1. Chat History
2. Retrieved Context
3. Web Search Context

If the answer exists in chat history,
use it.

If the answer exists in retrieved context,
use it.

If the answer cannot be found,
say you don't know.

Context:
{context}

Chat History:
{chat_history}

Question:
{input}
"""
    )


def get_llm(model_provider: str, model: str):

    if model_provider == "groq":
        return ChatGroq(
            model=model,
            api_key=GROQ_API_KEY,
            temperature=0
        )

    elif model_provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=GOOGLE_API_KEY,
            temperature=0
        )

    raise ValueError(f"Unsupported LLM Provider: {model_provider}")


class RAGChain:

    def __init__(self, llm, retriever, prompt):
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt

    def invoke(self, inputs):

        question = inputs["input"]

        docs = self.retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        prompt_value = self.prompt.invoke(
            {
                "context": context,
                "input": question
            }
        )

        response = self.llm.invoke(prompt_value)

        return {
            "answer": response.content
        }


def build_llm_chain(
    model_provider: str,
    model: str,
    vectorstore
):

    llm = get_llm(model_provider, model)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    prompt = get_prompt()

    return RAGChain(
        llm=llm,
        retriever=retriever,
        prompt=prompt
    )