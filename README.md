# RAG 연구 프로젝트

LangChain으로 구축된 RAG(Retrieval-Augmented Generation) 연구 프로젝트입니다. 이 프로젝트는 고급 문서 처리, 벡터 검색 및 챗봇 기능을 시연합니다. 주요 특징으로는 Magika를 사용한 AI 기반 파일 유형 탐지, FAISS를 위한 강력한 벡터 데이터베이스 래퍼, 그리고 여러 사용자 인터페이스(CLI, 문서 데모 UI, 챗봇 UI)가 있습니다.

## 🚀 핵심 아키텍처

### 문서 처리 파이프라인
-   **`module/document_parser.py`**: 파일 확장자에 의존하지 않고 Google의 Magika를 사용하여 지능적으로 파일 유형을 탐지하는 범용 문서 로더입니다. 다양한 형식을 지원하며 LangChain 로더와 통합됩니다.
-   **`module/vector_db.py`**: FAISS 벡터 저장소를 위한 래퍼입니다. 임베딩 호환성 검사, 선택적 영속성을 처리하고 벡터 연산을 위한 깔끔한 API를 제공합니다.

### RAG 챗 파이프라인
-   **`module/simple_rag_chat.py`**: 챗봇 애플리케이션의 핵심 RAG 로직을 구현합니다. BM25와 밀집 리트리버의 앙상블, 다중 쿼리 리트리버, 그리고 결과의 압축 및 재순위를 위한 Jina 리랭커를 포함한 고급 검색 체인을 특징으로 합니다. 또한 대화 기록을 관리합니다.

### 사용자 인터페이스
-   **`main.py`**: 문서 로딩 및 벡터 검색 기능을 시연하는 명령줄 인터페이스(CLI)입니다.
-   **`streamlit_app.py`**: 문서 업로드, 벡터 데이터베이스 관리 및 유사도 검색 수행을 위한 UI를 제공하는 Streamlit 웹 애플리케이션입니다.
-   **`chat_app.py`**: 벡터 데이터베이스의 문서를 기반으로 RAG 시스템과 대화할 수 있는 Streamlit 기반 챗봇 인터페이스입니다.

## 🛠️ 개발 명령어

### 환경 설정

```bash
# uv로 의존성 설치 (Python 3.12 필요)
uv sync

# API 키 구성을 위한 .env 파일 생성
cp .env.example .env
# 그 다음, .env 파일에 OpenAI API 키를 추가하세요
# OPENAI_API_KEY="your-key-here"
```

### 애플리케이션 실행

```bash
# 메인 CLI 데모 실행
uv run python main.py

# Streamlit 문서 처리 웹 앱 실행
uv run streamlit run streamlit_app.py

# Streamlit 챗봇 웹 앱 실행
uv run streamlit run chat_app.py
```

### 개별 모듈 테스트 실행

```bash
# 문서 파서 모듈 테스트
uv run python module/document_parser.py

# 벡터 데이터베이스 모듈 테스트
uv run python module/vector_db.py
```

## 📁 프로젝트 구조

```text
.
├── .claude/
├── .db/
├── .venv/
├── module/
│   ├── document_parser.py      # Magika AI 기반 문서 로더
│   ├── simple_rag_chat.py      # 핵심 RAG 챗봇 로직
│   └── vector_db.py            # FAISS 벡터 데이터베이스 래퍼
├── sample/
│   └── 국가별 공공부문 AI 도입 및 활용 전략.pdf # 샘플 문서
├── test/
│   ├── db/
│   └── faiss-test.py
├── .env.example
├── AGENTS.md
├── chat_app.py                 # Streamlit 챗봇 UI
├── CLAUDE.md
├── main.py                     # CLI 통합 데모
├── pyproject.toml
├── README.md
├── streamlit_app.py            # Streamlit 문서 데모 UI
└── uv.lock
```

## ✨ 주요 구현 상세

### 문서 로딩 (`document_parser.py`)
-   **AI 우선 파일 탐지**: 내용 기반 파일 유형 식별을 위해 `magika`에 의존합니다.
-   **광범위한 형식 지원**: PDF, TXT, CSV, HTML, DOCX, PPTX, XLSX, MD, JSON을 처리합니다.
-   **한국어 PDF 최적화**: 한국어 PDF에서 우수한 텍스트 추출을 위해 `PyMuPDFLoader`를 사용합니다.

### 벡터 데이터베이스 (`vector_db.py`)
-   **임베딩 호환성**: 디스크에서 데이터베이스를 로드할 때 차원 불일치 오류를 방지하기 위해 임베딩 모델 정보(`model_class`, `model_name`, `embedding_size`)를 저장하고 확인합니다.
-   **메모리 중심 및 영속성**: 속도를 위해 주로 메모리 내에서 작동하며, 재임베딩을 피하기 위해 디스크에 저장하는 옵션이 있습니다.

### RAG 챗 (`simple_rag_chat.py`)
-   **앙상블 리트리버**: 더 강력한 검색 결과를 위해 `BM25Retriever`(키워드 기반)와 `FAISS` 밀집 리트리버(의미 기반)를 결합합니다.
-   **쿼리 확장**: `MultiQueryRetriever`를 사용하여 사용자의 쿼리를 다른 관점에서 여러 변형으로 생성합니다.
-   **재순위**: `JinaRerank`를 사용하여 검색된 문서의 관련성을 재순위화하여 LLM에 제공되는 컨텍스트의 품질을 향상시킵니다.
-   **기록 관리**: `RunnableWithMessageHistory`를 사용하여 문맥 인식 응답을 위한 대화 기록을 유지합니다.

## 🔄 변경 로그

-   **2025-09-04**:
    -   현재 프로젝트 상태를 반영하기 위해 전체 문서를 철저히 업데이트하고 수정했습니다.
    -   프로젝트 구조 다이어그램과 파일 설명을 수정했습니다.
    -   `chat_app.py`와 `simple_rag_chat.py`에 대한 자세한 설명을 추가했습니다.
    -   개발 및 실행 명령어를 업데이트했습니다.
    -   이 변경 로그를 추가했습니다.
