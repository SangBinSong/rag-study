# RAG Study Project

AI 기반 문서 처리와 벡터 검색, 그리고 질의응답을 위한 RAG(Retrieval-Augmented Generation) 연구 및 데모 프로젝트입니다.

이 프로젝트는 최신 RAG 아키텍처의 핵심 구성 요소를 학습하고 실험할 수 있도록 설계되었습니다. 지능형 문서 로딩부터 고급 검색 기법(하이브리드 검색, 리랭킹) 및 대화형 UI에 이르는 전체 파이프라인을 포함합니다.

## 🚀 주요 특징

- **AI 기반 파일 검출**: Google의 Magika 라이브러리를 사용하여 파일 확장자가 아닌 내용 기반으로 문서를 식별합니다.
- **하이브리드 검색**: 의미적 유사도를 찾는 벡터 검색(FAISS)과 키워드 기반의 검색(BM25)을 결합하여 검색 정확도를 높입니다.
- **리랭킹**: Jina Reranker를 통해 검색된 결과의 순위를 재조정하여 가장 관련성 높은 문서를 상위로 올립니다.
- **세 가지 데모 앱**:
    1. `main.py`: 핵심 기능을 테스트하는 CLI 앱
    2. `streamlit_app.py`: 문서 로딩 및 벡터 DB 관리를 위한 웹 UI
    3. `chat_app.py`: 고급 RAG 파이프라인을 적용한 대화형 챗봇 UI
- **한국어 처리 최적화**: `PyMuPDFLoader`를 사용하여 한국어 PDF 문서에서 텍스트를 정확하게 추출합니다.
- **임베딩 호환성 관리**: 임베딩 모델 변경 시 벡터 DB의 호환성을 자동으로 검사하고 재구축을 유도합니다.

## 🛠️ 기술 스택

- **Python 3.12** & **uv**
- **LangChain**: RAG 파이프라인 구축
- **Streamlit**: 대화형 웹 UI
- **FAISS**: 고성능 벡터 유사도 검색 (Dense Retriever)
- **BM25**: 키워드 기반 검색 (Sparse Retriever)
- **Magika**: AI 기반 파일 타입 식별
- **Jina Rerank**: 검색 결과 리랭킹
- **OpenAI**: 텍스트 임베딩 및 LLM

## 📦 설치 및 실행

### 1. 환경 설정

```bash
# 프로젝트 복제
git clone https://github.com/your-username/rag-study.git
cd rag-study

# 의존성 설치 (uv 사용)
uv sync

# .env 파일 생성 및 API 키 설정
cp .env.example .env
```

`.env` 파일에 `OPENAI_API_KEY`를 입력하세요. Jina Reranker를 사용하려면 `JINA_API_KEY`도 필요할 수 있습니다.

### 2. 데모 실행

이 프로젝트는 세 가지 다른 데모 애플리케이션을 제공합니다.

#### 튜토리얼 1: 문서 로딩 및 DB 관리 (Web UI)

문서를 로딩하고 벡터 데이터베이스를 구축하는 UI입니다. **채팅 데모를 실행하기 전에 이 단계를 먼저 진행하는 것이 좋습니다.**

```bash
uv run streamlit run streamlit_app.py
```

#### 튜토리얼 2: RAG 채팅 (Web UI)

구축된 벡터 DB를 사용하여 문서 기반 질의응답을 수행하는 채팅 앱입니다.

```bash
uv run streamlit run chat_app.py
```

#### 튜토리얼 3: CLI 데모

주요 모듈의 핵심 기능을 테스트할 수 있는 커맨드 라인 인터페이스입니다.

```bash
uv run main.py
```

## 📁 프로젝트 구조

```text
rag-study/
├── module/               # 핵심 로직 모듈
│   ├── document_parser.py  # Magika AI 기반 지능형 문서 로더
│   └── vector_db.py        # FAISS 벡터 DB 래퍼 (호환성 검사 포함)
├── sample/               # 테스트용 샘플 문서
├── test/                 # 단위/통합 테스트 스크립트
├── .env.example          # 환경변수 템플릿
├── main.py               # 튜토리얼 3: CLI 데모
├── streamlit_app.py      # 튜토리얼 1: 문서 로딩 및 DB 관리 UI
├── chat_app.py           # 튜토리얼 2: RAG 채팅 UI
├── pyproject.toml        # 프로젝트 설정 및 의존성 (uv)
└── README.md             # 프로젝트 안내 문서
```

## 🔧 RAG 파이프라인 워크플로우 (`chat_app.py`)

`chat_app.py`는 다음과 같은 고급 RAG 파이프라인을 사용합니다.

1.  **사용자 질문** 입력
2.  **하이브리드 검색 (Ensemble Retriever)**:
    - **Dense Retriever (FAISS)**: 질문의 의미와 유사한 벡터를 검색합니다.
    - **Sparse Retriever (BM25)**: 질문의 핵심 키워드와 일치하는 문서를 검색합니다.
    - 두 검색 결과를 가중치(`[BM25: 0.4, FAISS: 0.6]`)를 두어 결합합니다.
3.  **리랭킹 (Contextual Compression)**:
    - **Jina Reranker**: 결합된 검색 결과 목록을 다시 평가하여 질문과 가장 관련성이 높은 순서로 정렬합니다.
4.  **프롬프트 생성**:
    - 리랭킹된 문서들을 컨텍스트로 하여 LLM에 전달할 프롬프트를 구성합니다.
5.  **LLM 응답 생성**:
    - **OpenAI (gpt-4.1-nano)** 모델이 컨텍스트를 기반으로 최종 답변을 생성합니다.
6.  **답변** 출력

## 🧪 테스트

개별 모듈 및 통합 기능을 테스트할 수 있습니다.

```bash
# 모듈별 기능 테스트
uv run module/document_parser.py
uv run module/vector_db.py

# 통합 테스트
uv run test/faiss-test.py
uv run test/langchain-test.py
```

## 📚 학습 목표

- AI 기반 파일 타입 식별 (Magika)
- 하이브리드 검색 (FAISS + BM25) 및 리랭킹 (Jina)을 포함한 고급 RAG 아키텍처 이해
- LangChain을 활용한 복잡한 RAG 파이프라인 구축
- 임베딩 모델 호환성 관리 및 벡터 DB 운영
- Streamlit을 이용한 인터랙티브 데모 앱 개발

## 🤝 기여

이슈나 개선사항이 있으면 언제든 제안해 주세요!

## 📄 라이선스

MIT License
