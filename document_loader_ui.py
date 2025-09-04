"""
문서 로더 웹 UI 테스트

이 스크립트는 PDF 문서를 로드하고 분석하며 벡터 데이터베이스에 저장하는 프로세스를
Streamlit을 통한 웹 인터페이스로 테스트합니다.
"""

import tempfile
import time
import streamlit as st
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv, find_dotenv, set_key, get_key

from module.document_parser import load_documents
from module.vector_db import VectorDB
from langchain.schema import Document

# 환경 변수 로드
load_dotenv(find_dotenv(), override=True)


# 페이지 설정
st.set_page_config(
    page_title="문서 로더 테스트",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화ㅁ
if 'documents' not in st.session_state:
    st.session_state.documents = None
if 'stats' not in st.session_state:
    st.session_state.stats = {}
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
# DB 경로 고정
st.session_state.db_path = "./db/faiss"


def init_vector_db(force_new=False):
    """벡터 데이터베이스 초기화
    
    Args:
        force_new: 강제로 새 인스턴스 생성 여부
    """
    if force_new or st.session_state.vector_db is None:
        # 기존 객체 제거
        if 'vector_db' in st.session_state:
            del st.session_state['vector_db']
            
        # 새 인스턴스 생성 (force_new 옵션 전달)
        st.session_state.vector_db = VectorDB(
            storage_path=st.session_state.db_path,
            force_new=force_new  # 이 값이 True면 디스크에 파일이 있어도 무시하고 새로 만듬
        )
        
    return st.session_state.vector_db


def display_document_stats(docs: List[Document], stats: Dict[str, Any]):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("원본 문서", stats.get('original_document_count', len(docs)))
    with col2:
        st.metric("청크 문서", stats.get('chunked_document_count', len(docs)))
    with col3:
        st.metric("총 문자 수", f"{stats.get('total_characters', 0):,}")
    with col4:
        st.metric("평균 문서 길이", f"{stats.get('average_doc_length', 0):,} 문자")
    
    # 파일 유형 통계
    st.subheader("📑 파일 유형")
    detected_types = stats.get('detected_types', {})
    if detected_types:
        st.json(detected_types)
    else:
        st.info("파일 유형 정보 없음")
    
    # 이미지 및 표 통계
    image_count = 0
    image_saved_count = 0
    table_count = 0
    
    for doc in docs:
        images = doc.metadata.get("images", [])
        tables = doc.metadata.get("tables", [])
        image_count += len(images)
        
        # 실제로 저장된 이미지만 카운트
        for img in images:
            if "image_path" in img and Path(img["image_path"]).exists():
                image_saved_count += 1
                
        table_count += len(tables)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("추출된 이미지 총수", image_count)
    with col2:
        st.metric("저장된 이미지 파일", image_saved_count)
    with col3:
        st.metric("추출된 표", table_count)


def display_document_preview(docs: List[Document], limit: int = 3):
    """문서 미리보기 표시"""
    st.subheader("👁️ 문서 미리보기")
    
    # 미리볼 문서 수 선택
    preview_limit = min(limit, len(docs))
    
    for i, doc in enumerate(docs[:preview_limit]):
        with st.expander(f"문서 {i+1}/{preview_limit}", expanded=i==0):
            # 내용
            st.markdown("**📄 내용:**")
            st.text_area(
                label=f"문서 내용",
                value=doc.page_content[:3000] + ("..." if len(doc.page_content) > 3000 else ""),
                height=200,
                key=f"doc_content_{i}"
            )
            
            # 메타데이터
            st.markdown("**ℹ️ 메타데이터:**")
            meta_preview = {k: v for k, v in doc.metadata.items() 
                           if k not in ["images", "tables"]}
            st.json(meta_preview)
            
            # 이미지 정보
            images = doc.metadata.get("images", [])
            if images:
                st.markdown(f"**🖼️ 이미지 ({len(images)}개):**")
                for j, img in enumerate(images[:3]):
                    cols = st.columns([1, 2])
                    with cols[0]:
                        st.markdown(f"**이미지 {j+1}:**")
                        st.write(f"크기: {img.get('width')}x{img.get('height')}")
                        st.write(f"형식: {img.get('image_format', 'unknown')}")
                    
                    with cols[1]:
                        image_path = img.get("image_path")
                        if image_path and Path(image_path).exists():
                            st.image(image_path, width=300)
                        else:
                            st.info("이미지 파일이 저장되지 않았습니다.")
                        
                        ocr_text = img.get("ocr_text")
                        if ocr_text:
                            with st.expander("OCR 텍스트"):
                                st.text(ocr_text)
                
                if len(images) > 3:
                    st.info(f"... 외 {len(images) - 3}개 이미지 (모두 표시하지 않음)")
            
            # 표 정보
            tables = doc.metadata.get("tables", [])
            if tables:
                st.markdown(f"**📊 표 ({len(tables)}개):**")
                for j, table in enumerate(tables[:3]):
                    with st.expander(f"표 {j+1}: {table.get('rows')}행 x {table.get('columns')}열"):
                        # 표 데이터를 Streamlit 테이블로 표시
                        if "data" in table:
                            st.table(table["data"])
                        else:
                            st.text(table.get("text", "표 데이터 없음"))
                
                if len(tables) > 3:
                    st.info(f"... 외 {len(tables) - 3}개 표 (모두 표시하지 않음)")


def main():
    st.title("📚 문서 로더 테스트")

    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # OpenAI API Key
        openai_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            value=get_key(find_dotenv(), 'OPENAI_API_KEY') or '',
            help="임베딩 생성을 위한 OpenAI API 키"
        )
        if openai_key:
            set_key(find_dotenv(), 'OPENAI_API_KEY', openai_key)
        
        # 청크 설정
        st.subheader("📝 청크 설정")
        chunk_size = st.slider("청크 크기", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("청크 오버랩", 50, 500, 200, 50)
        
        # PDF 처리 기본 설정 (사용자 UI 없음)
        extract_images = True
        extract_tables = True
        split_documents = True
        
        image_dir = None  # 이미지 파일 저장 안함
        
        st.divider()
        
        # 벡터 DB 상태
        st.subheader("🗂️ 벡터 DB 상태")
        if st.button("🔄 상태 새로고침"):
            vector_db = init_vector_db()
            stats = vector_db.get_stats()
            st.json(stats)
    
    # OpenAI API 키 확인
    if not get_key(find_dotenv(), 'OPENAI_API_KEY'):
        st.warning("⚠️ OpenAI API 키를 사이드바에서 설정해주세요!")
    
    # 메인 탭
    tab1, tab2, tab3 = st.tabs(["📄 문서 로딩", "🔍 문서 미리보기", "🗂️ 벡터 DB 저장"])
    
    with tab1:
        st.header("📄 문서 로딩")
        
        # 파일 업로드 옵션
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📁 파일 업로드")
            uploaded_files = st.file_uploader(
                "문서 파일을 선택하세요",
                type=['pdf', 'txt', 'docx', 'pptx', 'xlsx', 'csv', 'md', 'html', 'json'],
                help="Magika AI가 파일 내용을 분석하여 타입을 결정합니다",
                accept_multiple_files=True
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
        
        # 처리할 파일 결정
        files_to_process = []
        
        if uploaded_files:
            # 업로드된 파일을 임시 저장
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    files_to_process.append({
                        'path': tmp_file.name,
                        'name': uploaded_file.name,
                        'temp': True
                    })
            
            st.info(f"📤 업로드됨: {len(files_to_process)}개 파일")
            for i, file_info in enumerate(files_to_process):
                st.text(f"  {i+1}. {file_info['name']}")
        
        elif hasattr(st.session_state, 'selected_file'):
            files_to_process.append({
                'path': st.session_state.selected_file,
                'name': Path(st.session_state.selected_file).name,
                'temp': False
            })
        
        if files_to_process and st.button("🚀 문서 로딩 실행", type="primary"):
            all_documents = []
            all_stats = {
                'original_document_count': 0,
                'chunked_document_count': 0,
                'total_characters': 0,
                'detected_types': {},
                'total_sources': 0
            }
            
            # 각 문서별 통계를 저장할 변수
            individual_results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                for i, file_info in enumerate(files_to_process):
                    status_text.text(f"📊 {i+1}/{len(files_to_process)} - {file_info['name']} 처리 중...")
                    
                    with st.spinner(f"📄 {file_info['name']} 분석 중..."):
                        documents, stats = load_documents(
                            path=file_info['path'],
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            split_documents=split_documents,
                            extract_images=extract_images,
                            extract_tables=extract_tables,
                            image_dir=image_dir if extract_images else None
                        )
                        
                        # 개별 문서 통계 저장 (평균 문서 길이를 청크 문서 기준으로 계산)
                        if stats.get('chunked_document_count', 0) > 0:
                            stats['average_doc_length'] = stats.get('total_characters', 0) // stats.get('chunked_document_count', 1)
                        else:
                            stats['average_doc_length'] = 0
                            
                        individual_results.append({
                            'file_name': file_info['name'],
                            'documents': documents,
                            'stats': stats
                        })
                        
                        # 통계 병합
                        all_documents.extend(documents)
                        all_stats['original_document_count'] += stats.get('original_document_count', 0)
                        all_stats['chunked_document_count'] += stats.get('chunked_document_count', 0)
                        all_stats['total_characters'] += stats.get('total_characters', 0)
                        all_stats['total_sources'] += 1
                        
                        # 파일 유형 병합
                        for dtype, count in stats.get('detected_types', {}).items():
                            if dtype in all_stats['detected_types']:
                                all_stats['detected_types'][dtype] += count
                            else:
                                all_stats['detected_types'][dtype] = count
                    
                    # 진행 상황 업데이트
                    progress_bar.progress((i + 1) / len(files_to_process))
                
                # 평균 문서 길이 계산
                if all_stats['chunked_document_count'] > 0:
                    all_stats['average_doc_length'] = all_stats['total_characters'] // all_stats['chunked_document_count']
                else:
                    all_stats['average_doc_length'] = 0
                    
                # 세션 상태에 저장
                st.session_state.documents = all_documents
                st.session_state.stats = all_stats
                st.session_state.individual_results = individual_results
                
                # 결과 표시
                status_text.text("✅ 처리 완료!")
                progress_bar.progress(100)
                st.success(f"✅ 성공! {len(all_documents)}개 문서 처리 완료")
                
                # 전체 통계
                st.subheader("📊 전체 통계")
                display_document_stats(all_documents, all_stats)
                
                # 각 문서별 통계
                st.subheader("📂 각 문서별 통계")
                for idx, result in enumerate(individual_results):
                    with st.expander(f"{idx+1}. {result['file_name']}", expanded=(idx==0)):
                        display_document_stats(result['documents'], result['stats'])
                
            except Exception as e:
                st.error(f"❌ 문서 로딩 실패: {str(e)}")
            
            finally:
                # 임시 파일 정리
                for file_info in files_to_process:
                    if file_info.get('temp', False) and Path(file_info['path']).exists():
                        try:
                            Path(file_info['path']).unlink(missing_ok=True)
                        except:
                            pass
    
    with tab2:
        st.header("🔍 문서 미리보기")
        
        if st.session_state.documents:
            documents = st.session_state.documents
            
            # 미리볼 문서 수 선택
            preview_count = st.slider("미리볼 문서 수", 1, min(10, len(documents)), 3)
            
            # 문서 미리보기 표시
            display_document_preview(documents, limit=preview_count)
            
        else:
            st.info("먼저 '문서 로딩' 탭에서 문서를 로드해주세요.")
    
    with tab3:
        st.header("🗂️ 벡터 데이터베이스 저장")
        
        # 벡터 DB 초기화
        vector_db = init_vector_db()
        
        # 현재 DB 상태 표시
        col1, col2 = st.columns([4, 1])
        with col1:
            st.subheader("📊 현재 상태")
        with col2:
            if st.button("🔄 새로고침", key="refresh_status"):
                st.rerun()
                
        stats = vector_db.get_stats()
        st.json(stats)
        
        is_empty = vector_db.is_empty()
        if is_empty:
            st.info("📭 벡터 데이터베이스가 비어있습니다")
        else:
            st.success("📚 벡터 데이터베이스에 데이터가 있습니다")
        
        # 문서 추가
        st.subheader("🔧 작업")
        
        if st.session_state.documents:
            documents = st.session_state.documents
            
            if st.button("📝 문서 추가", type="primary"):
                try:
                    with st.spinner("벡터 임베딩 생성 중..."):
                        vector_db.add_documents(documents)
                    
                    st.success(f"✅ {len(documents)}개 문서가 추가되었습니다!")
                    
                    # 저장
                    with st.spinner("디스크에 저장 중..."):
                        vector_db.save()
                    st.success("💾 벡터 데이터베이스가 디스크에 저장되었습니다!")
                    
                    # 2초 후 자동 새로고침
                    with st.spinner("화면 갱신 중..."):
                        time.sleep(1)
                        st.rerun()
                    
                    # 업데이트된 상태
                    st.subheader("📊 업데이트된 상태")
                    new_stats = vector_db.get_stats()
                    st.json(new_stats)
                    
                except Exception as e:
                    st.error(f"❌ 벡터 DB 작업 실패: {str(e)}")
        else:
            st.warning("⚠️ 먼저 '문서 로딩' 탭에서 문서를 로드해주세요.")
        
        # 데이터베이스 초기화
        st.markdown("---")
        
        if st.button("🔄 데이터베이스 초기화", key="reset_db_button", type="secondary"):
            try:
                # force_new=True로 새 인스턴스 생성
                init_vector_db(force_new=True)
                
                # 성공 메시지 표시 
                st.success("🔄 벡터 데이터베이스가 초기화되었습니다.")
                
                # 새로고침
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"벡터 데이터베이스 초기화 오류: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.error(f"벡터 데이터베이스 초기화 오류: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
