# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) study project built with LangChain. It demonstrates advanced document processing, vector search, and chat capabilities. The key features include AI-based file type detection with Magika, a robust vector database wrapper for FAISS, and multiple user interfaces (CLI, Document Demo UI, Chat UI).

## Core Architecture

### Document Processing Pipeline
-   **`module/document_parser.py`**: A universal document loader that uses Google's Magika for intelligent file type detection, independent of file extensions. It supports various formats and integrates with LangChain loaders.
-   **`module/vector_db.py`**: A wrapper for the FAISS vector store. It handles embedding compatibility checks, selective persistence, and provides a clean API for vector operations.

### RAG Chat Pipeline
-   **`module/simple_rag_chat.py`**: Implements the core RAG logic for the chat application. It features an advanced retrieval chain, including an ensemble of BM25 and dense retrievers, a multi-query retriever, and a Jina re-ranker for compressing and re-ranking results. It also manages chat history.

### User Interfaces
-   **`main.py`**: A command-line interface (CLI) to demonstrate document loading and vector search functionalities.
-   **`streamlit_app.py`**: A Streamlit web application that provides a UI for uploading documents, managing the vector database, and performing similarity searches.
-   **`chat_app.py`**: A Streamlit-based chat interface that allows users to have a conversation with the RAG system based on the documents in the vector database.

## Development Commands

### Environment Setup

```bash
# Install dependencies with uv (Python 3.12 required)
uv sync

# Create .env file for API key configuration
cp .env.example .env
# Then, add your OpenAI API key to the .env file
# OPENAI_API_KEY="your-key-here"
```

### Running the Applications

```bash
# Run the main CLI demo
uv run python main.py

# Run the Streamlit document processing web app
uv run streamlit run streamlit_app.py

# Run the Streamlit chat web app
uv run streamlit run chat_app.py
```

### Running Individual Modules for Testing

```bash
# Test the document parser module
uv run python module/document_parser.py

# Test the vector database module
uv run python module/vector_db.py
```

## Project Structure

```text
.
├── .claude/
├── .db/
├── .venv/
├── module/
│   ├── document_parser.py      # Magika AI-based document loader
│   ├── simple_rag_chat.py      # Core RAG chat logic
│   └── vector_db.py            # FAISS vector database wrapper
├── sample/
│   └── 국가별 공공부문 AI 도입 및 활용 전략.pdf # Sample document
├── test/
│   ├── db/
│   └── faiss-test.py
├── .env.example
├── AGENTS.md
├── chat_app.py                 # Streamlit Chat UI
├── CLAUDE.md
├── main.py                     # CLI integration demo
├── pyproject.toml
├── README.md
├── streamlit_app.py            # Streamlit Document Demo UI
└── uv.lock
```

## Critical Implementation Details

### Document Loading (`document_parser.py`)
-   **AI-First File Detection**: Relies on `magika` for content-based file type identification.
-   **Broad Format Support**: Handles PDF, TXT, CSV, HTML, DOCX, PPTX, XLSX, MD, and JSON.
-   **Korean PDF Optimization**: Uses `PyMuPDFLoader` for superior text extraction from Korean PDFs.

### Vector Database (`vector_db.py`)
-   **Embedding Compatibility**: Saves and checks embedding model information (`model_class`, `model_name`, `embedding_size`) to prevent dimension conflicts when loading a database from disk.
-   **Memory-Centric with Persistence**: Operates primarily in-memory for speed, with an option to save to disk to avoid re-embedding.

### RAG Chat (`simple_rag_chat.py`)
-   **Ensemble Retriever**: Combines `BM25Retriever` (keyword-based) and a `FAISS` dense retriever (semantic-based) for more robust search results.
-   **Query Expansion**: Uses `MultiQueryRetriever` to generate multiple variations of a user's query from different perspectives.
-   **Re-ranking**: Employs `JinaRerank` to re-rank the retrieved documents for relevance, improving the quality of the context provided to the LLM.
-   **History Management**: `RunnableWithMessageHistory` is used to maintain conversation history for context-aware responses.

## Changelog

-   **2025-09-04**:
    -   Thoroughly updated and revised the entire document to reflect the current project state.
    -   Corrected the project structure diagram and file descriptions.
    -   Added detailed explanations for `chat_app.py` and `simple_rag_chat.py`.
    -   Updated development and execution commands.
    -   Added this changelog.
