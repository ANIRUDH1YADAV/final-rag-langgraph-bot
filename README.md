 Advanced RAG with LangGraph
Corrective Retrieval-Augmented Generation (CRAG)

Unlike traditional RAG systems that immediately generate an answer from retrieved documents, this chatbot first evaluates the quality of each retrieved document using an LLM.

Depending on the retrieval quality, the workflow follows one of three execution paths:

Correct Retrieval – Answer is generated directly from highly relevant documents.
Ambiguous Retrieval – Moderately relevant documents are combined with web search results before answer generation.
Incorrect Retrieval – The query is rewritten, web search is performed, and the answer is generated from refined web information.

This adaptive workflow improves answer quality and reduces hallucinations.
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41.1-red.svg)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.60-green.svg)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.13-blue.svg)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.23-purple.svg)](https://www.trychroma.com/)
[![Code style: Python](https://img.shields.io/badge/Code%20Style-Python-black.svg)](https://www.python.org/dev/peps/pep-0008/)
## ✨ Features

-   Upload and chat with PDF documents
-   Corrective RAG (CRAG)
-   LLM-based document grading
-   Automatic query rewriting
-   Tavily web search fallback
-   Knowledge refinement
-   Groq and Gemini support
-   Chroma vector database
-   FastAPI backend
-   Streamlit frontend

## 🏗️ Workflow

``` text
User Question
    ↓
Retrieve Documents
    ↓
LLM Document Grading
    ├── CORRECT (>0.7) → Refine → Generate
    ├── AMBIGUOUS (0.3–0.7) → Good Docs + Web Search → Refine → Generate
    └── INCORRECT (<0.3) → Rewrite → Web Search → Refine → Generate
```

## 🛠️ Tech Stack

-   FastAPI
-   LangGraph
-   LangChain
-   Streamlit
-   Groq
-   Google Gemini
-   ChromaDB
-   HuggingFace Embeddings
-   Tavily Search

## 📂 Project Structure

``` text
client/
server/
├── api/
├── config/
├── core/
│   ├── document_processor.py
│   ├── vector_database.py
│   ├── llm_chain_factory.py
│   └── langgraph/
│       ├── graph.py
│       ├── nodes.py
│       └── state.py
└── main.py
```

## ⚙️ Installation

``` bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 🔑 Environment Variables

``` env
GROQ_API_KEY=
GOOGLE_API_KEY=
TAVILY_API_KEY=
HF_TOKEN=
```

## ▶️ Run

Backend:

``` bash
cd server
uvicorn main:app --reload
```

Frontend:

``` bash
cd client
streamlit run app.py
```

## 🧪 Example Questions

-   What challenges of AI in healthcare are mentioned?
-   Explain the benefits and limitations of AI in healthcare.
-   Who won the FIFA World Cup 2022?

## 👨‍💻 Author

**Anirudh Yadav**

AI / Machine Learning Engineer

## 📄 License

MIT License.
