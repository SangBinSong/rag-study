import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple

import fitz  # PyMuPDF
import streamlit as st
from PIL import Image, ImageDraw

from new_document_loader import DocumentLoader, CHUNK_SIZE, CHUNK_OVERLAP, DB_DIR

# 이미지 렌더링 DPI
DEFAULT_DPI = 300


# =========================
# 페이지 이미지 및 오버레이 렌더링
# =========================
@st.cache_data(show_spinner=False)
def render_page_image(pdf_path: str, page: int, dpi: int = DEFAULT_DPI) -> Image.Image:
    """PDF 페이지를 이미지로 렌더링"""
    doc = fitz.open(pdf_path)
    try:
        p = doc[page - 1]
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = p.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img
    finally:
        doc.close()


def bbox_pt_to_imgpx(bbox_pt: Tuple[float, float, float, float], dpi: int = DEFAULT_DPI):
    """PyMuPDF 좌표(pt) → 렌더 이미지 픽셀 변환"""
    x0, y0, x1, y1 = bbox_pt
    scale = dpi / 72
    return (x0 * scale, y0 * scale, x1 * scale, y1 * scale)


def draw_overlays(base: Image.Image, chunks: List[Dict[str, Any]], dpi: int = DEFAULT_DPI,
                  highlight_id: str = "") -> Image.Image:
    """청크 하이라이트를 이미지에 오버레이"""
    img = base.copy()
    draw = ImageDraw.Draw(img)
    # 청크 타입별 색상 설정
    color_map = {"text": (0, 128, 255), "table": (255, 128, 0), "image": (0, 200, 120)}

    for ch in chunks:
        b = ch["bbox"]
        # 바운딩 박스 좌표 변환
        x0, y0, x1, y1 = bbox_pt_to_imgpx((b["x0"], b["y0"], b["x1"], b["y1"]), dpi)
        color = color_map.get(ch["type"], (0, 255, 0))
        width = 4 if ch["id"] == highlight_id else 2  # 선택된 청크는 더 두꺼운 선
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)

    return img


def get_pdf_info(pdf_path: str) -> Dict[str, Any]:
    """PDF 파일의 기본 정보를 추출"""
    doc = fitz.open(pdf_path)
    try:
        info = {
            "page_count": len(doc),
            "file_name": Path(pdf_path).name,
            "file_size": Path(pdf_path).stat().st_size,
            "metadata": doc.metadata
        }
        return info
    finally:
        doc.close()


# =========================
# Streamlit UI
# =========================
def main():
    st.set_page_config(page_title="문서 로더 UI", layout="wide")
    st.title("📄 문서 로더 (RAG 벡터 DB 구축)")

    # 사이드바 설정
    st.sidebar.header("설정")

    chunk_size = st.sidebar.number_input(
        "청크 크기",
        value=CHUNK_SIZE,
        min_value=100,
        max_value=2000,
        step=50,
        help="청크당 최대 단어 수"
    )

    chunk_overlap = st.sidebar.number_input(
        "청크 오버랩",
        value=CHUNK_OVERLAP,
        min_value=0,
        max_value=chunk_size // 2,
        step=10,
        help="연속 청크 간 겹치는 단어 수"
    )

    index_name = st.sidebar.text_input(
        "인덱스 이름",
        value="index",
        help="저장될 FAISS 인덱스 이름"
    )

    st.sidebar.markdown("---")

    # 문서 업로드 섹션
    st.header("PDF 문서 업로드 및 처리")
    uploaded_file = st.file_uploader("PDF 파일을 선택하세요", type=["pdf"])

    # 임시 저장된 파일 경로
    pdf_path = None

    if uploaded_file:
        # 업로드된 파일을 임시 위치에 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        # PDF 정보 표시
        pdf_info = get_pdf_info(pdf_path)
        st.success(f"파일 업로드 완료: {uploaded_file.name}")

        col1, col2 = st.columns(2)
        col1.metric("페이지 수", f"{pdf_info['page_count']} 페이지")
        col2.metric("파일 크기", f"{pdf_info['file_size'] / 1024 / 1024:.2f} MB")

        # 문서 분석 및 자동 저장 버튼
        if st.button("문서 분석 및 벡터 DB 저장", use_container_width=True):
            # 로더 인스턴스 생성
            loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            # 로딩 표시 및 처리
            with st.spinner("문서를 분석 중입니다..."):
                # [수정됨] 원본 파일명을 load_document에 전달
                original_filename = uploaded_file.name
                chunks = loader.load_document(pdf_path, original_filename=original_filename)

                if not chunks:
                    st.error("문서 처리 중 오류가 발생했습니다.")
                else:
                    st.session_state['chunks'] = chunks
                    st.session_state['pdf_path'] = pdf_path
                    st.success(f"문서 분석 완료: {len(chunks)}개 청크 생성")

                    # 페이지별로 청크 그룹화
                    pages = sorted({c.page for c in chunks})
                    st.session_state['pages'] = pages

                    # 첫 페이지를 기본 선택
                    if 'current_page' not in st.session_state:
                        st.session_state['current_page'] = pages[0] if pages else 1

                    # 벡터 DB에 자동 저장
                    with st.spinner("임베딩 생성 및 FAISS 인덱스 저장 중..."):
                        success = loader.save_to_faiss(chunks, index_name)
                        if success:
                            st.success(f"FAISS 인덱스 저장 완료: {os.path.join(DB_DIR, f'{index_name}.faiss')}")

                            # 저장된 정보 표시
                            col1, col2, col3 = st.columns(3)
                            col1.metric("총 청크 수", len(chunks))

                            # 이미지를 제외한 임베딩 청크 수
                            embed_chunks = [c for c in chunks if c.type != "image"]
                            col2.metric("임베딩 청크 수", len(embed_chunks))

                            # 이미지 청크 수
                            img_chunks = [c for c in chunks if c.type == "image"]
                            col3.metric("이미지 청크 수", len(img_chunks))
                        else:
                            st.error("벡터 DB 저장 중 오류가 발생했습니다.")

        # 분석된 청크가 있으면 미리보기 표시
        if 'chunks' in st.session_state:
            st.header("PDF 청크 미리보기")

            # 페이지 선택기
            page_selector = st.select_slider(
                "페이지 선택",
                options=st.session_state['pages'],
                value=st.session_state['current_page']
            )
            st.session_state['current_page'] = page_selector

            # 현재 페이지의 청크 필터링
            current_page_chunks = [c for c in st.session_state['chunks'] if c.page == page_selector]

            # 청크 타입 필터
            chunk_types = st.multiselect(
                "청크 타입 필터링",
                ["text", "table", "image"],
                default=["text", "table", "image"]
            )

            filtered_chunks = [c for c in current_page_chunks if c.type in chunk_types]

            # 페이지 이미지 렌더링
            try:
                col1, col2 = st.columns([3, 2])

                with col1:
                    st.subheader(f"페이지 {page_selector} 미리보기")
                    base_img = render_page_image(st.session_state['pdf_path'], page_selector)

                    # 청크 선택기
                    chunk_ids = [c.id for c in filtered_chunks]
                    selected_chunk = None

                    if chunk_ids:
                        selected_id = st.selectbox("청크 선택", chunk_ids)
                        selected_chunk = next((c for c in filtered_chunks if c.id == selected_id), None)

                    # 오버레이 이미지 생성
                    chunks_dict = [asdict(c) for c in filtered_chunks]
                    img = draw_overlays(base_img, chunks_dict, highlight_id=selected_id if selected_chunk else "")

                    # 이미지 표시
                    st.image(img, caption=f"페이지 {page_selector}의 청크 시각화", use_container_width=True)
                    st.caption("파란색: text, 주황색: table, 초록색: image (선 두께 ↑ = 선택된 청크)")

                # 청크 상세 정보 표시
                with col2:
                    if selected_chunk:
                        st.subheader("청크 상세 정보")
                        st.markdown(
                            f"**Type**: `{selected_chunk.type}`  \n"
                            f"**Page**: `{selected_chunk.page}`  \n"
                            f"**Order**: `{selected_chunk.order}`  \n"
                            f"**Parent ID**: `{selected_chunk.parent_object_id or ''}`  \n"
                            f"**Hash**: `{selected_chunk.hash[:10]}...`  \n"
                        )

                        st.write("**좌표 정보**")
                        st.json(selected_chunk.bbox)
                        st.write("**정규화된 좌표**")
                        st.json(selected_chunk.nbbox)

                        # 이미지 청크에 대한 메타데이터 표시 개선
                        if selected_chunk.type == "image" and hasattr(selected_chunk,
                                                                      'metadata') and selected_chunk.metadata:
                            st.write("**이미지 메타데이터**")
                            st.json(selected_chunk.metadata)

                            # 이미지 크기 분류 표시
                            if "size_category" in selected_chunk.metadata:
                                st.info(f"이미지 크기 분류: {selected_chunk.metadata['size_category']}")

                        st.write("**내용 미리보기**")
                        if selected_chunk.type == "text":
                            st.text_area("텍스트", selected_chunk.content[:1000], height=200)
                        elif selected_chunk.type == "table":
                            st.text_area("CSV 데이터", selected_chunk.content[:1000], height=200)
                        elif selected_chunk.type == "image":
                            if selected_chunk.image_path and os.path.exists(selected_chunk.image_path):
                                st.image(selected_chunk.image_path, caption="추출된 이미지")

                                # 관련 텍스트 청크가 있는지 표시
                                if hasattr(selected_chunk,
                                           'related_text_chunks') and selected_chunk.related_text_chunks:
                                    st.write("**관련 텍스트 청크**")
                                    for rel in selected_chunk.related_text_chunks:
                                        st.markdown(
                                            f"- 관련성 점수: `{rel['relevance_score']:.2f}` - {rel['preview'][:100]}...")

            except Exception as e:
                st.error(f"미리보기 생성 중 오류가 발생했습니다: {str(e)}")

    else:
        st.info("PDF 파일을 업로드해주세요.")

    # 정리: 임시 파일 삭제
    if st.button("세션 정리"):
        if 'pdf_path' in st.session_state and os.path.exists(st.session_state['pdf_path']):
            try:
                os.unlink(st.session_state['pdf_path'])
                st.session_state.pop('pdf_path', None)
                st.session_state.pop('chunks', None)
                st.session_state.pop('pages', None)
                st.session_state.pop('current_page', None)
                st.success("세션 정리 완료")
                st.rerun()
            except Exception as e:
                st.error(f"임시 파일 삭제 중 오류: {str(e)}")


if __name__ == "__main__":
    main()
