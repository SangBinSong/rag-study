"""
RAG Study Streamlit App - Web UI for Document Loading and Vector Search
"""

import streamlit as st
import os
from pathlib import Path
import tempfile
from module.document_parser import load_documents
from module.vector_db import VectorDB

# Page config
st.set_page_config(
    page_title="RAG Study Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = None
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'db_path' not in st.session_state:
    st.session_state.db_path = "./db/streamlit_rag_demo"

def init_vector_db():
    """Initialize vector database"""
    if st.session_state.vector_db is None:
        st.session_state.vector_db = VectorDB(storage_path=st.session_state.db_path)
    return st.session_state.vector_db

def main():
    st.title("🔍 RAG Study Demo")
    st.markdown("**Magika AI 기반 문서 로딩 + FAISS 벡터 검색**")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # OpenAI API Key
        openai_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            value=os.getenv('OPENAI_API_KEY', ''),
            help="임베딩 생성을 위한 OpenAI API 키"
        )
        if openai_key:
            os.environ['OPENAI_API_KEY'] = openai_key
        
        # Chunk settings
        st.subheader("📝 청크 설정")
        chunk_size = st.slider("청크 크기", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("청크 오버랩", 50, 500, 200, 50)
        
        st.divider()
        
        # Vector DB info
        st.subheader("🗂️ 벡터 DB 상태")
        if st.button("🔄 상태 새로고침"):
            if st.session_state.vector_db:
                stats = st.session_state.vector_db.get_stats()
                st.json(stats)
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        st.warning("⚠️ OpenAI API 키를 사이드바에서 설정해주세요!")
        return
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📄 문서 로딩", "🗂️ 벡터 데이터베이스", "🔍 검색"])
    
    with tab1:
        st.header("📄 문서 로딩")
        
        # File upload options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📁 파일 업로드")
            uploaded_file = st.file_uploader(
                "문서 파일을 선택하세요",
                type=['pdf', 'txt', 'docx', 'pptx', 'xlsx', 'csv', 'md', 'html', 'json'],
                help="Magika AI가 파일 내용을 분석하여 타입을 결정합니다"
            )
        
        with col2:
            st.subheader("📂 샘플 파일")
            sample_path = "sample/국가별 공공부문 AI 도입 및 활용 전략.pdf"
            if Path(sample_path).exists():
                if st.button("🇰🇷 샘플 PDF 사용", type="primary"):
                    st.session_state.selected_file = sample_path
                    st.success(f"샘플 파일 선택됨: {sample_path}")
            else:
                st.error(f"샘플 파일을 찾을 수 없습니다: {sample_path}")
        
        # Process file
        file_to_process = None
        
        if uploaded_file:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_to_process = tmp_file.name
                st.info(f"📤 업로드됨: {uploaded_file.name}")
        
        elif hasattr(st.session_state, 'selected_file'):
            file_to_process = st.session_state.selected_file
        
        if file_to_process and st.button("🚀 문서 로딩 실행", type="primary"):
            try:
                with st.spinner("📊 Magika AI로 문서 분석 중..."):
                    documents = load_documents(
                        file_to_process, 
                        chunk_size=chunk_size, 
                        chunk_overlap=chunk_overlap,
                        split_documents=True
                    )
                    
                st.session_state.documents = documents
                
                # Show results
                st.success(f"✅ 성공! {len(documents)}개 청크로 분할됨")
                
                # Document statistics
                if documents:
                    # Calculate statistics
                    total_chars = sum(len(doc.page_content) for doc in documents)
                    detected_types = {}
                    detection_methods = {'magika': 0}
                    
                    for doc in documents:
                        detected_type = doc.metadata.get('detected_type', 'unknown')
                        detected_types[detected_type] = detected_types.get(detected_type, 0) + 1
                        detection_methods['magika'] += 1
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("총 문서", len(documents))
                    with col2:
                        st.metric("총 문자 수", f"{total_chars:,}")
                    with col3:
                        st.metric("평균 청크 길이", f"{total_chars // len(documents):,}")
                    with col4:
                        st.metric("소스 파일", len(set(doc.metadata.get('source', 'unknown') for doc in documents)))
                    
                    # File type detection
                    st.subheader("🔍 Magika 검출 결과")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.json({"검출된 타입": detected_types})
                    with col2:
                        st.json({"검출 방식": detection_methods})
                    
                    # Sample content
                    with st.expander("📖 첫 번째 청크 미리보기"):
                        first_doc = documents[0]
                        st.text(first_doc.page_content[:500] + "..." if len(first_doc.page_content) > 500 else first_doc.page_content)
                        st.json(first_doc.metadata)
                        
            except Exception as e:
                st.error(f"❌ 문서 로딩 실패: {str(e)}")
            
            finally:
                # Clean up temporary file
                if uploaded_file and file_to_process and Path(file_to_process).exists():
                    try:
                        os.unlink(file_to_process)
                    except:
                        pass
    
    with tab2:
        st.header("🗂️ 벡터 데이터베이스")
        
        # Initialize vector DB
        vector_db = init_vector_db()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 현재 상태")
            stats = vector_db.get_stats()
            st.json(stats)
            
            is_empty = vector_db.is_empty()
            if is_empty:
                st.info("📭 벡터 데이터베이스가 비어있습니다")
            else:
                st.success("📚 벡터 데이터베이스에 데이터가 있습니다")
        
        with col2:
            st.subheader("🔧 작업")
            
            # Add documents
            if st.session_state.documents:
                if st.button("📝 문서 추가", type="primary"):
                    try:
                        with st.spinner("벡터 임베딩 생성 중..."):
                            # Add first 20 chunks for demo
                            docs_to_add = st.session_state.documents[:20]
                            vector_db.add_documents(docs_to_add)
                        
                        st.success(f"✅ {len(docs_to_add)}개 문서 청크가 추가되었습니다!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 문서 추가 실패: {str(e)}")
            else:
                st.warning("⚠️ 먼저 문서를 로딩해주세요")
            
            # Save to disk
            if not is_empty:
                if st.button("💾 디스크에 저장"):
                    try:
                        vector_db.save()
                        st.success("✅ 벡터 데이터베이스가 디스크에 저장되었습니다!")
                    except Exception as e:
                        st.error(f"❌ 저장 실패: {str(e)}")
            
            # Clear database
            if st.button("🗑️ 데이터베이스 초기화", type="secondary"):
                st.session_state.vector_db = None
                st.success("🔄 데이터베이스가 초기화되었습니다!")
                st.rerun()
    
    with tab3:
        st.header("🔍 유사도 검색")
        
        vector_db = init_vector_db()
        
        if vector_db.is_empty():
            st.warning("⚠️ 먼저 문서를 벡터 데이터베이스에 추가해주세요!")
            return
        
        # Initialize query in session state
        if 'search_query' not in st.session_state:
            st.session_state.search_query = ""
        
        # Predefined queries
        st.subheader("💡 예시 질문")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("AI 도입 전략"):
                st.session_state.search_query = "AI 도입 전략은 무엇인가?"
        with col2:
            if st.button("공공부문 인공지능"):
                st.session_state.search_query = "공공부문에서의 인공지능 활용"
        with col3:
            if st.button("디지털 전환"):
                st.session_state.search_query = "디지털 전환과 혁신"
        
        # Search interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "🔍 검색할 내용을 입력하세요",
                value=st.session_state.search_query,
                placeholder="예: AI 도입 전략은 무엇인가요?",
                key="query_input"
            )
            # Update session state when user types
            if query != st.session_state.search_query:
                st.session_state.search_query = query
        
        with col2:
            k = st.selectbox("결과 개수", [3, 5, 10], index=1)
        
        # Search execution
        if query and st.button("🚀 검색 실행", type="primary"):
            try:
                with st.spinner("🔍 유사한 문서 검색 중..."):
                    results = vector_db.similarity_search(query, k=k)
                
                if results:
                    st.success(f"✅ {len(results)}개의 관련 문서를 찾았습니다!")
                    
                    for i, doc in enumerate(results, 1):
                        with st.expander(f"📄 결과 {i}", expanded=i <= 3):
                            # Content
                            st.markdown("**내용:**")
                            st.text(doc.page_content)
                            
                            # Metadata
                            st.markdown("**메타데이터:**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("파일명", doc.metadata.get('file_name', 'N/A'))
                            with col2:
                                st.metric("검출 타입", doc.metadata.get('detected_type', 'N/A'))
                            with col3:
                                st.metric("파일 크기", f"{doc.metadata.get('file_size', 0):,} bytes")
                else:
                    st.warning("🤔 관련 문서를 찾을 수 없습니다.")
                    
            except Exception as e:
                st.error(f"❌ 검색 실패: {str(e)}")

if __name__ == "__main__":
    main()