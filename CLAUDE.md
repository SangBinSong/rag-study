# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) study project built with LangChain, focusing on AI-based document processing and vector search capabilities. The project demonstrates modern document loading with Magika AI file detection and FAISS-based vector storage.

## Core Architecture

### Document Processing Pipeline
- **Document Loading**: `module/document-load.py` provides universal document loading using Magika AI for intelligent file type detection (no extension dependency)
- **Vector Storage**: `module/vector-db.py` wraps FAISS with embedding compatibility checks and optional persistence
- **Integration Point**: `main.py` serves as the entry point for demonstrations and testing

### Key Design Principles
- **AI-First File Detection**: Uses Magika library instead of file extensions for accurate document type identification
- **Memory-Centric Vector Storage**: FAISS operates primarily in-memory with selective persistence to avoid re-embedding
- **Embedding Compatibility**: Strict embedding model validation prevents vector dimension mismatches

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
# Run main demo (CLI)
uv run main.py

# Run Streamlit web app
uv run streamlit run streamlit_app.py

# Test document loader with sample PDF
uv run module/document-parser.py

# Test vector database functionality  
uv run module/vector-db.py
```

### Testing (simplified)

```bash
# Run individual test files
uv run test/faiss-test.py
uv run test/langchain-test.py
uv run test/pydantic-test.py
```

## Project Structure

```text
rag-study/
├── module/
│   ├── document-parser.py  # Magika AI-based document loader
│   └── vector-db.py        # FAISS vector database wrapper
├── test/                   # Simple test scripts
│   ├── faiss-test.py
│   ├── langchain-test.py
│   └── pydantic-test.py
├── sample/                 # Sample documents
│   └── 국가별 공공부문 AI 도입 및 활용 전략.pdf
├── main.py                 # CLI integration demo
├── streamlit_app.py        # Web UI demo
├── .env.example            # Environment variables template
├── pyproject.toml          # uv project configuration
└── CLAUDE.md              # This file
```

## Module Integration

### DocumentLoader + VectorDB Workflow

```python
from module.document_parser import load_documents
from module.vector_db import VectorDB

# Load documents with Magika AI detection
documents = load_documents(
    "sample/국가별 공공부문 AI 도입 및 활용 전략.pdf",
    chunk_size=1000, 
    chunk_overlap=200,
    split_documents=True
)

# Store in vector database
vector_db = VectorDB(storage_path="./db/faiss_store")
vector_db.add_documents(documents)
results = vector_db.similarity_search("AI 도입 전략", k=5)
```

## Critical Implementation Details

### Document Loading Pipeline

- **Magika Detection**: Document loading relies entirely on AI-based content analysis, not file extensions
- **Supported Formats**: PDF, TXT, CSV, HTML, DOCX, PPTX, XLSX, MD, JSON through content analysis
- **Korean PDF Support**: Uses PyMuPDFLoader for better Korean text extraction from PDFs
- **Chunk Management**: Documents are automatically split into manageable chunks with configurable overlap

### Vector Database Architecture

- **Embedding Compatibility**: Vector database automatically saves/loads embedding model info to prevent dimension conflicts
- **Memory-Centric Design**: FAISS operates primarily in-memory with selective persistence to avoid re-embedding
- **Persistence Strategy**: Only saves to disk when explicitly requested, preventing unnecessary I/O
- **Compatibility Checks**: Strict validation of embedding dimensions and model types

### Environment Requirements

- **Python Version**: Requires exactly Python 3.12 (specified in pyproject.toml)
- **OpenAI API**: Required for text embeddings (text-embedding-3-small model)
- **UV Package Manager**: All dependencies managed through uv for reproducible builds

## Sample Data

- `sample/국가별 공공부문 AI 도입 및 활용 전략.pdf` - Korean government AI adoption strategy document for testing

## Key Dependencies

Core packages: `langchain`, `langchain-community`, `langchain-openai`, `langchain-pymupdf4llm`, `faiss-cpu`, `magika`, `streamlit`
