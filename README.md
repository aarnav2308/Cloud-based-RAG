Cloud-Based Retrieval-Augmented Generation (RAG) System

=========================================================

Project Overview: 

This project demonstrates an end-to-end Retrieval-Augmented Generation (RAG) system built using cloud services, vector databases, and local large language models. The system retrieves relevant information from domain-specific documents using semantic search and generates grounded, context-aware answers.

The architecture and implementation reflect real-world production RAG systems commonly used in enterprise AI and cloud-based applications.

A) Key Capabilities:

1) Document embedding and semantic indexing

2) Vector-based semantic similarity search

3) Context-aware answer generation

4) Local LLM inference for privacy and cost control

5) Source attribution with similarity scores

6) Modular, production-oriented Python architecture

B) System Workflow:

1) User submits a natural language query

2) Query is converted into a vector embedding

3) Embedding is compared against stored document vectors

4) Top-K relevant document chunks are retrieved

5) Retrieved context is assembled into a prompt

6) A local LLM generates a grounded response

7) Final answer is returned with source references

8) A local LLM generates a grounded response

9) Final answer is returned with source references

C) Screenshots and Validation:

The screenshots/ directory contains execution evidence demonstrating:

1) Pinecone index creation and index statistics

2) Successful embedding generation

3) Context retrieval with similarity scores

4) Ollama model availability and usage

5) End-to-end RAG query execution

D) How to Run the Project:
1) Install Dependencies
pip install -r requirements.txt

2) Start Ollama
ollama serve


3) Ensure required models are available:

ollama pull mistral
ollama pull nomic-embed-text

4) Run a Sample Query
python -c "from rag.rag_chain import rag_answer; print(rag_answer('What is ecofeminism?'))"

E) Description of main py files:

ingest.py: Processes PDFs, chunks text, generates embeddings, and stores vectors in the database

ollama_client.py: Interfaces with local Ollama models for embeddings and response generation

query_retrieval.py: Converts user queries to embeddings and retrieves top-K similar document chunks

lambda_function.py: Wraps the RAG pipeline into a serverless AWS Lambda-compatible API
