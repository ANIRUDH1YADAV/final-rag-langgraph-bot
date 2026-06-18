RAG PDFBot

This project is a Retrieval-Augmented Generation (RAG) chatbot that allows users to query multiple PDFs for contextual answers. It integrates LangGraph for flexible workflow orchestration, maintains chat memory, rewrites queries for better retrieval, and uses Chroma DB for document search.

Features
RAG System: Combines document retrieval with an LLM to generate grounded answers.
LangGraph Workflow: Modular nodes like retrieval, grading, web search, and answer generation.
Chat Memory: Keeps track of user-LLM interactions for contextual follow-ups.
Query Rewriting: Improves vague or ambiguous questions for more accurate retrieval.
Chroma DB: Vector database for storing and searching document embeddings.
Architecture Overview
Document Ingestion:
PDFs are uploaded and split into text chunks.
Each chunk is embedded into vectors and stored in Chroma DB.
Query Flow:
User question is rewritten if needed.
Closest document chunks are retrieved from Chroma DB.
Context is fed to the LLM to generate a grounded response.
Memory is updated for follow-up queries.
Setup Instructions

Clone the Repository:

git clone https://github.com/YourUsername/RAG-PDFBot.git
cd RAG-PDFBot

Install Dependencies:

pip install -r requirements.txt

Run the Application:

uvicorn main:app --reload
Access:
The FastAPI app will run locally at http://127.0.0.1:8000.
API docs can be viewed at /docs.
Example Usage
Upload PDFs via the /upload_and_process_pdfs endpoint.
Use the /chat endpoint to ask questions about the content.
Follow-up questions will use chat memory for context.
Tech Stack
FastAPI: Web framework.
LangGraph: Manages multi-step reasoning.
Chroma DB: Vector storage for document embeddings.
LLMs: Models like LLaMA or Gemini for text generation.
Future Enhancements
Pinecone integration for scalable vector search.
