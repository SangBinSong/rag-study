# RAG Study Project

AI 기반 문서 처리와 벡터 검색을 위한 RAG(Retrieval-Augmented Generation) 연구 프로젝트입니다.

## 🚀 주요 특징

- **Magika AI 파일 검출**: 확장자에 의존하지 않는 AI 기반 문서 타입 식별
- **FAISS 벡터 검색**: 메모리 기반 고성능 유사도 검색
- **한국어 문서 지원**: PyMuPDFLoader를 통한 정확한 한국어 PDF 처리
- **임베딩 호환성**: 모델 변경 시 자동 호환성 검사 및 재구축

## 🛠️ 기술 스택

- **Python 3.12**: 최신 Python 기능 활용
- **LangChain**: 문서 로딩 및 텍스트 분할
- **FAISS**: 고성능 벡터 유사도 검색
- **Magika**: Google의 AI 기반 파일 타입 검출
- **OpenAI Embeddings**: 텍스트 임베딩 생성
- **uv**: 빠른 Python 패키지 관리

## 📦 설치 및 실행

### 환경 설정

```bash
# 의존성 설치
uv sync

# OpenAI API 키 설정
export OPENAI_API_KEY="your-api-key-here"
```

### 데모 실행

```bash
# CLI 데모 실행
uv run main.py

# Streamlit 웹 앱 실행
uv run streamlit run streamlit_app.py

# 개별 모듈 테스트
uv run module/document-load.py  # 문서 로더 테스트
uv run module/vector-db.py      # 벡터 데이터베이스 테스트
```

## 📁 프로젝트 구조

```text
rag-study/
├── module/
│   ├── document-load.py    # Magika AI 기반 문서 로더
│   └── vector-db.py        # FAISS 벡터 데이터베이스 래퍼
├── sample/
│   └── 국가별 공공부문 AI 도입 및 활용 전략.pdf  # 테스트용 한국어 PDF
├── test/                   # 간단한 테스트 스크립트들
├── main.py                 # 통합 데모
└── CLAUDE.md              # Claude Code용 개발 가이드
```

## 🔧 사용법

### 기본 워크플로우

```python
from module.document_load import DocumentLoader
from module.vector_db import VectorDB

# 1. 문서 로딩 (Magika AI 검출)
loader = DocumentLoader(chunk_size=1000, chunk_overlap=200)
documents = loader.load_and_split("sample/국가별 공공부문 AI 도입 및 활용 전략.pdf")

# 2. 벡터 데이터베이스 구축
vector_db = VectorDB(storage_path="./db/faiss_store")
vector_db.add_documents(documents)

# 3. 유사도 검색
results = vector_db.similarity_search("AI 도입 전략은 무엇인가?", k=5)
for doc in results:
    print(f"답변: {doc.page_content[:100]}...")
```

### 지원 파일 형식

Magika AI가 내용을 분석하여 지원하는 형식:

- PDF, DOCX, PPTX, XLSX
- TXT, MD, HTML, CSV, JSON
- 확장자에 관계없이 파일 내용으로 판단

## ⚡ 성능 최적화

### 메모리 중심 설계

- FAISS 인메모리 연산으로 빠른 검색
- 선택적 디스크 저장으로 재임베딩 방지
- 임베딩 모델 호환성 자동 검증

### 한국어 처리 최적화

- PyMuPDFLoader로 한국어 PDF 정확도 향상
- 적절한 청크 크기와 오버랩으로 컨텍스트 보존

## 🧪 테스트

```bash
# 개별 컴포넌트 테스트
uv run test/faiss-test.py      # FAISS 기능 테스트
uv run test/langchain-test.py  # LangChain 통합 테스트
uv run test/pydantic-test.py   # 데이터 모델 테스트
```

## 📚 학습 목표

이 프로젝트를 통해 학습할 수 있는 내용:

1. **현대적 문서 처리**: AI 기반 파일 타입 검출
2. **벡터 데이터베이스**: FAISS를 활용한 고성능 검색
3. **RAG 아키텍처**: 문서 검색과 생성 모델의 결합
4. **임베딩 관리**: 모델 호환성과 버전 관리
5. **한국어 NLP**: 다국어 문서 처리 최적화

## 🔍 데모 결과 예시

```text
🔍 RAG Study Demo - AI 기반 문서 로딩 + 벡터 검색
============================================================
📋 지원 포맷: ['pdf', 'txt', 'csv', 'html', 'docx', 'pptx', 'xlsx', 'markdown', 'json']

📄 문서 로딩: sample/국가별 공공부문 AI 도입 및 활용 전략.pdf
✅ 성공! 47개 청크로 분할됨
📊 문서 통계:
   - 총 문자 수: 45,821
   - 평균 청크 길이: 974
   - 검출된 파일 타입: {'pdf': 47}
   - 검출 방식: {'magika': 47}

🔍 유사도 검색 데모:
1. 질문: AI 도입 전략은 무엇인가?
   답변 1: 정부는 공공부문 AI 도입을 위한 종합적인 전략을 수립하여...
   답변 2: 디지털 정부 혁신을 통해 국민 서비스 향상과 행정 효율성을...
```

## 🤝 기여

이슈나 개선사항이 있으면 언제든 제안해 주세요!

## 📄 라이선스

MIT License
