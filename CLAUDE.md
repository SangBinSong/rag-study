# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) study project built with LangChain. It showcases a complete RAG pipeline, from intelligent document loading to advanced retrieval and chat interaction. The project serves as a comprehensive demonstration and educational resource for modern RAG architectures.

There are three main applications:
1.  **CLI Demo (`main.py`)**: A command-line interface to test document loading and basic vector search.
2.  **Web UI Demo (`streamlit_app.py`)**: A Streamlit web application for interactively loading documents and managing the vector database.
3.  **Chat UI (`chat_app.py`)**: A Streamlit-based chat interface demonstrating an advanced RAG pipeline with hybrid search and reranking.

## Core Architecture

### Document Processing Pipeline
- **Document Loading (`module/document_parser.py`)**: Provides universal document loading using Magika AI for intelligent file type detection, independent of file extensions.
- **Vector Storage (`module/vector_db.py`)**: A wrapper for FAISS that includes embedding model compatibility checks and persistent storage.

### RAG Pipelines
- **Basic RAG (`streamlit_app.py`, `main.py`)**: Uses a standard vector store for similarity search.
- **Advanced RAG (`chat_app.py`)**: Implements a more sophisticated retrieval process:
    - **Hybrid Search**: Combines dense (FAISS) and sparse (BM25) retrievers using an `EnsembleRetriever`.
    - **Reranking**: Uses `JinaRerank` via a `ContextualCompressionRetriever` to improve the relevance of retrieved documents before sending them to the LLM.

## Development Commands

### Environment Setup

```bash
# Install dependencies with uv (Python 3.12 required)
uv sync

# Create .env file for API key configuration
cp .env.example .env
# Or create directly
echo "OPENAI_API_KEY=your-key-here" > .env
```

### Running the Project

```bash
# Run the main CLI demo
uv run main.py

# Run the Streamlit web app for document management
uv run streamlit run streamlit_app.py

# Run the Streamlit chat application
uv run streamlit run chat_app.py
```

### Testing

```bash
# Run individual module tests
uv run module/document_parser.py
uv run module/vector_db.py

# Run integration test scripts
uv run test/faiss-test.py
uv run test/langchain-test.py
uv run test/pydantic-test.py
```

## Project Structure

```text
rag-study/
├── .claude/              # Claude AI settings
├── db/                   # Default directory for FAISS vector stores
│   ├── faiss/
│   └── streamlit_rag_demo/
├── module/               # Core reusable modules
│   ├── document_parser.py  # Magika AI-based document loader
│   └── vector_db.py        # FAISS vector database wrapper
├── sample/               # Sample documents for testing
│   └── 국가별 공공부문 AI 도입 및 활용 전략.pdf
├── test/                 # Test scripts
│   ├── db/
│   ├── faiss-test.py
│   ├── langchain-test.py
│   └── pydantic-test.py
├── .env.example          # Environment variables template
├── chat_app.py           # Advanced RAG chat application (Streamlit)
├── main.py               # CLI integration demo
├── streamlit_app.py      # Web UI for document loading/DB management
├── pyproject.toml        # uv project configuration and dependencies
├── CLAUDE.md             # This file: guidance for the Claude agent
└── README.md             # General project documentation for developers
```

## Critical Implementation Details

### Document Loading (`document_parser.py`)
- **Magika Detection**: Document loading relies on AI-based content analysis, not file extensions.
- **Korean PDF Support**: Uses `PyMuPDFLoader` for robust Korean text extraction.
- **Chunk Management**: Automatically splits documents into manageable chunks.

### Vector Database (`vector_db.py`)
- **Embedding Compatibility**: Automatically saves and validates the embedding model (`text-embedding-3-small`) to prevent dimension conflicts.
- **Persistence**: FAISS index is operated in-memory but can be saved to disk.

### Chat Pipeline (`chat_app.py`)
- **Ensemble Retriever**: Combines BM25 (keyword-based) and FAISS (semantic) search results for improved retrieval quality. `weights` are set to `[0.4, 0.6]` for BM25 and FAISS respectively.
- **Jina Reranker**: The `ContextualCompressionRetriever` uses `jina-reranker-m0` to re-order the retrieved documents, pushing the most relevant ones to the top.
- **Prompt Engineering**: Documents are formatted into an XML-like structure (`<documents>...`) within the system prompt for the LLM.

## Environment Requirements
- **Python Version**: Python 3.12
- **Package Manager**: `uv`
- **API Keys**: `OPENAI_API_KEY` is required for embeddings and chat. `JINA_API_KEY` may be required for the reranker.
