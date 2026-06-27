# 🤖 CRAG PDFBot

**Corrective Retrieval-Augmented Generation (CRAG) chatbot for PDF
documents**

A production-ready AI chatbot that answers questions from uploaded PDF
documents using a **Corrective Retrieval-Augmented Generation (CRAG)**
pipeline built with **LangGraph**, **FastAPI**, and **Streamlit**.

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
