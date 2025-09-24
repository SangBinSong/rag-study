import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import faiss
import fitz
import numpy as np
import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI

try:
    import pytesseract
    from PIL import Image

    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

############# CONFIG #############
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
DB_DIR = "./db/faiss"
IMAGE_DIR = "./extracted_images"
IMAGE_KEYWORDS = ['그림', '이미지', '도표', '차트', '스크린샷', '화면', '다이어그램', '구조도', 'figure', 'image', 'chart', 'diagram', 'screenshot']
############# CONFIG #############


@dataclass
class Chunk:
    id: str
    type: str
    content: str
    page: int
    bbox: Dict[str, float]
    nbbox: Dict[str, float]
    order: int
    parent_object_id: Optional[str]
    source: Dict[str, Any]
    hash: str
    image_path: Optional[str] = None
    caption: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    related_text_chunks: Optional[List[Dict[str, Any]]] = None


def merge_text_blocks(blocks: List[Tuple[Tuple[float, float, float, float], str]],
                      separation_threshold: float = 5.0, 
                      max_merge_size: int = 2000) -> List[Tuple[Tuple[float, float, float, float], str]]:
    """
    시각적으로 가까운 텍스트 블록을 병합합니다.
    - blocks: (bbox, text) 튜플의 리스트
    - separation_threshold: 이 값(pt)보다 세로 간격이 크면 다른 문단으로 취급 (기본값을 5.0으로 줄임)
    - max_merge_size: 병합된 텍스트의 최대 길이 (문자 수 기준)
    """
    if not blocks:
        return []

    # 위에서 아래로, 왼쪽에서 오른쪽으로 정렬
    blocks.sort(key=lambda b: (b[0][1], b[0][0]))

    merged_blocks = []
    current_bbox, current_text = blocks[0]

    for i in range(1, len(blocks)):
        next_bbox, next_text = blocks[i]

        # 이전 블록의 하단과 다음 블록의 상단 사이의 수직 거리
        vertical_gap = next_bbox[1] - current_bbox[3]
        
        # 수평 겹침 확인 (더 엄격한 조건)
        horizontal_overlap = not (next_bbox[0] > current_bbox[2] or next_bbox[2] < current_bbox[0])
        
        # 병합 조건을 더 엄격하게 설정
        should_merge = (
            vertical_gap < separation_threshold and  # 수직 거리가 임계값보다 작고
            horizontal_overlap and  # 수평으로 겹치고
            len(current_text) + len(next_text) < max_merge_size  # 최대 크기 제한
        )

        if should_merge:
            # 텍스트 병합
            current_text += "\n" + next_text
            # 바운딩 박스 확장
            x0 = min(current_bbox[0], next_bbox[0])
            y0 = min(current_bbox[1], next_bbox[1])
            x1 = max(current_bbox[2], next_bbox[2])
            y1 = max(current_bbox[3], next_bbox[3])
            current_bbox = (x0, y0, x1, y1)
        else:
            # 병합 종료, 현재까지의 블록을 리스트에 추가하고 새 블록 시작
            merged_blocks.append((current_bbox, current_text))
            current_bbox, current_text = next_bbox, next_text

    # 마지막으로 처리 중이던 블록 추가
    merged_blocks.append((current_bbox, current_text))

    return merged_blocks

def ensure_dirs():
    Path(DB_DIR).mkdir(parents=True, exist_ok=True)
    Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)

def sha256_of(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def normalize_bbox(bbox: Tuple[float, float, float, float], w: float, h: float) -> Dict[str, float]:
    x0, y0, x1, y1 = bbox
    return {"x0": x0 / w, "y0": y0 / h, "x1": x1 / w, "y1": y1 / h}

def _categorize_image_size(width: int, height: int) -> str:
    total_pixels = width * height
    if total_pixels < 10000: return "작은이미지" # 100x100 이하
    if total_pixels < 100000: return "중간이미지" # 316x316 이하
    if total_pixels < 500000: return "큰이미지" # 707x707 이하
    return "매우큰이미지"

def _find_related_text_chunks(image_info: Dict[str, Any], text_chunks: List[Chunk]) -> List[Dict[str, Any]]:
    if not text_chunks: return []
    related = []
    image_page = image_info.get('page', 0)
    page_chunks = [chunk for chunk in text_chunks if chunk.page == image_page]
    for chunk in page_chunks:
        relevance_score = 0.0
        content = chunk.content.lower()
        for keyword in IMAGE_KEYWORDS:
            if keyword in content:
                relevance_score += 0.3
        image_refs = re.findall(r'(그림|이미지|도표|차트|figure|image)\s*(\d+)', content)
        for _, ref_num in image_refs:
            if int(ref_num) == image_info.get('order', 0):
                relevance_score += 0.5
        if relevance_score > 0.3:
            related.append({'chunk_id': chunk.id, 'relevance_score': relevance_score, 'preview': chunk.content[:100],
                            'page': chunk.page})
    return sorted(related, key=lambda x: x['relevance_score'], reverse=True)[:3]


def hierarchical_chunking(text: str, max_chunk_size: int, overlap_size: int) -> List[str]:
    """
    계층형 청크 나누기
    """
    if not text.strip():
        return []

    sentences = re.split(r'(?<=[.?!。！？])\s+', text.strip()) # 문장 단위로 쪼개기
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text]  # 문장 구분이 안되면 원본 텍스트 반환

    chunks = []
    current_chunk_words = []

    for sentence in sentences:
        sentence_words = sentence.split()
        if len(current_chunk_words) + len(sentence_words) > max_chunk_size and current_chunk_words:
            chunks.append(" ".join(current_chunk_words))
            overlap_word_count = 0
            overlap_sentences = []
            for sent in reversed(chunks[-1].split(".")):
                words = (sent + ".").split()
                if overlap_word_count + len(words) > overlap_size: break
                overlap_sentences.insert(0, sent + ".")
                overlap_word_count += len(words)
            current_chunk_words = " ".join(overlap_sentences).split() + sentence_words
        else:
            current_chunk_words.extend(sentence_words)

    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))

    return chunks


def is_scanned_page(page: fitz.Page) -> bool:
    blocks = page.get_text("blocks")
    text_len = sum(len(b[4] or "") for b in blocks if len(b) >= 5)
    has_images = len(page.get_images(full=True)) > 0
    return (text_len < 20) and has_images


def ocr_page_text(page: fitz.Page, dpi: int = 200) -> Optional[str]:
    if not OCR_AVAILABLE: return None
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(img, lang="kor+eng").strip() or None


def extract_text_blocks(page: fitz.Page) -> List[Tuple[Tuple[float, float, float, float], str]]:
    results = []
    for b in page.get_text("blocks", sort=True):  # sort=True로 읽기 순서 보장
        if len(b) >= 5 and (b[4] or "").strip():
            results.append(((float(b[0]), float(b[1]), float(b[2]), float(b[3])), b[4].strip()))
    return results


def extract_tables_with_bbox(pdf_path: str, page_num_zero_based: int) -> List[
    Tuple[Tuple[float, float, float, float], List[List[str]]]]:
    tables_out = []
    with pdfplumber.open(pdf_path) as pdf:
        if page_num_zero_based < len(pdf.pages):
            page = pdf.pages[page_num_zero_based]
            for t in page.find_tables():
                if t.extract() and any(any(cell for cell in row) for row in t.extract()):
                    tables_out.append((t.bbox, t.extract()))
    return tables_out


def table_to_csv_text(rows: List[List[str]]) -> str:
    return "\n".join([",".join(f"\"{(c or '').replace('\"', '\"\"')}\"" for c in r) for r in rows])


def extract_images(page: fitz.Page, out_dir: str, pdf_name: str) -> List[
    Dict[str, Any]]:
    results = []
    page_num = page.number + 1
    for img_idx, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        if xref <= 0: continue
        try:
            rects = page.get_image_rects(xref)
            bbox = rects[0].irect if rects else (page.rect * 0.25).irect
            pix = fitz.Pixmap(page.parent, xref)
            if pix.width < 20 or pix.height < 20: continue
            if pix.colorspace.name not in (fitz.csRGB.name, fitz.csGRAY.name):
                pix = fitz.Pixmap(fitz.csRGB, pix)

            fname = f"{pdf_name}_p{page_num:04d}_img{img_idx:03d}.png"
            fpath = str(Path(out_dir) / fname)
            pix.save(fpath)

            results.append({
                "path": fpath, "bbox": tuple(bbox), "width": pix.width, "height": pix.height,
                "size_category": _categorize_image_size(pix.width, pix.height),
                "order": img_idx + 1, "page": page_num
            })
        except Exception as e:
            logger.warning(f"페이지 {page_num} 이미지 {img_idx} 처리 실패: {e}")
    return results


def embed_texts(texts: List[str], client: OpenAI, model: str = EMBED_MODEL, batch: int = 1024) -> np.ndarray:
    if not texts: return np.zeros((0, EMBED_DIM), dtype="float32")
    vecs = []
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=model, input=texts[i:i + batch])
        vecs.append(np.array([d.embedding for d in resp.data], dtype="float32"))
    return np.vstack(vecs)


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, eps)

# PDF 처리 메인 파이프라인
def process_pdf(pdf_path: str, doc_id_prefix: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP,
                original_filename: str = None) -> List[Chunk]:
    doc_chunks, text_chunks, image_chunks = [], [], []
    doc_path = Path(pdf_path)

    pdf_name = Path(original_filename).stem if original_filename else doc_path.stem

    with fitz.open(pdf_path) as doc:
        for pno, page in enumerate(doc):
            page_num = pno + 1
            w, h = page.rect.width, page.rect.height
            imgs = extract_images(page, IMAGE_DIR, pdf_name)
            for i_order, it in enumerate(imgs, start=1):
                parent_id = f"{doc_id_prefix}_p{page_num:04d}_obj_image{i_order:04d}"
                image_chunks.append(Chunk(
                    id=f"{parent_id}_chunk001", type="image", content="", page=page_num,
                    bbox={"x0": it["bbox"][0], "y0": it["bbox"][1], "x1": it["bbox"][2], "y1": it["bbox"][3],
                          "unit": "pt"},
                    nbbox=normalize_bbox(it["bbox"], w, h), order=1, parent_object_id=parent_id,
                    source={"pdf_path": str(doc_path.resolve()), "page_width": w, "page_height": h},
                    hash=sha256_of(it["path"]), image_path=it["path"], metadata=it
                ))

            # 텍스트 블록 추출 후 병합 과정 추가
            raw_blocks = extract_text_blocks(page)

            # 병합된 블록을 사용
            merged_blocks = merge_text_blocks(raw_blocks, separation_threshold=5.0, max_merge_size=chunk_size * 4)

            if not merged_blocks and is_scanned_page(page):
                ocr_text = ocr_page_text(page)
                if ocr_text: merged_blocks = [((0.0, 0.0, w, h), ocr_text)]

            for b_order, (bbox, text) in enumerate(merged_blocks, start=1):
                parent_id = f"{doc_id_prefix}_p{page_num:04d}_obj_text{b_order:04d}"
                chunks = hierarchical_chunking(text, chunk_size, overlap)
                for i, c in enumerate(chunks, start=1):
                    text_chunks.append(Chunk(
                        id=f"{parent_id}_chunk{i:03d}", type="text", content=c, page=page_num,
                        bbox={"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3], "unit": "pt"},
                        nbbox=normalize_bbox(bbox, w, h), order=i, parent_object_id=parent_id,
                        source={"pdf_path": str(doc_path.resolve()), "page_width": w, "page_height": h},
                        hash=sha256_of(c)
                    ))

            for t_order, (tbbox, rows) in enumerate(extract_tables_with_bbox(pdf_path, pno), start=1):
                csv_text = table_to_csv_text(rows)
                parent_id = f"{doc_id_prefix}_p{page_num:04d}_obj_table{t_order:04d}"
                bbox = (float(tbbox[0]), float(tbbox[1]), float(tbbox[2]), float(tbbox[3]))
                text_chunks.append(Chunk(
                    id=f"{parent_id}_chunk001", type="table", content=csv_text, page=page_num,
                    bbox={"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3], "unit": "pt"},
                    nbbox=normalize_bbox(bbox, w, h), order=1, parent_object_id=parent_id,
                    source={"pdf_path": str(doc_path.resolve()), "page_width": w, "page_height": h},
                    hash=sha256_of(csv_text)
                ))

    for img_chunk in image_chunks:
        img_chunk.related_text_chunks = _find_related_text_chunks(img_chunk.metadata, text_chunks)

    doc_chunks.extend(text_chunks)
    doc_chunks.extend(image_chunks)
    return doc_chunks


# 문서 로더 클래스
class DocumentLoader:
    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, exclude_image_data=True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.exclude_image_data = exclude_image_data
        ensure_dirs()
        self.client = OpenAI()

    def load_document(self, pdf_path: str, original_filename: str = None) -> List[Chunk]:
        logger.info(f"Processing: {original_filename or pdf_path}")
        doc_id_prefix = sha256_of(str(Path(pdf_path).resolve()))[:18]
        try:
            chunks = process_pdf(pdf_path, doc_id_prefix, self.chunk_size, self.chunk_overlap, original_filename)
            logger.info(f"Generated {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}", exc_info=True)
            return []

    def create_embeddings(self, chunks: List[Chunk]) -> Tuple[np.ndarray, List[str]]:
        texts_to_embed = [c.content for c in chunks if c.type != "image"]
        ids = [c.id for c in chunks if c.type != "image"]
        if not texts_to_embed:
            return np.zeros((0, EMBED_DIM), dtype="float32"), []
        logger.info(f"Embedding {len(texts_to_embed)} chunks...")
        vecs = embed_texts(texts_to_embed, self.client, model=EMBED_MODEL)
        return l2_normalize(vecs), ids

    def save_to_faiss(self, chunks: List[Chunk], index_name: str = "index", original_filename: str = None) -> bool:
        if not chunks: return False
        vecs, ids = self.create_embeddings(chunks)
        if len(vecs) == 0: return False

        # LangChain 호환 FAISS 벡터스토어 생성
        from langchain_community.vectorstores import FAISS
        from langchain_community.docstore.in_memory import InMemoryDocstore
        from langchain_core.documents import Document
        
        # Document 객체 생성
        documents = []
        # 원본 파일명 추출
        if not original_filename and chunks:
            # 첫 번째 청크의 source에서 파일 경로 추출
            first_chunk_source = chunks[0].source
            if isinstance(first_chunk_source, dict) and 'pdf_path' in first_chunk_source:
                pdf_path = first_chunk_source['pdf_path']
                original_filename = Path(pdf_path).name
            else:
                original_filename = "문서"
        
        for i, chunk in enumerate(chunks):
            if chunk.type != "image":  # 이미지가 아닌 청크만
                doc = Document(
                    page_content=chunk.content,
                    metadata={
                        "id": chunk.id,
                        "type": chunk.type,
                        "page": chunk.page,
                        "order": chunk.order,
                        "parent_object_id": chunk.parent_object_id,
                        "source": chunk.source,
                        "hash": chunk.hash,
                        "bbox": chunk.bbox,
                        "nbbox": chunk.nbbox,
                        "file_name": original_filename  # 파일명 추가
                    }
                )
                documents.append(doc)
        
        # FAISS 벡터스토어 생성 및 저장 (LangChain OpenAIEmbeddings 사용)
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        # LangChain 형식으로 저장
        vectorstore.save_local(DB_DIR)
        
        # 추가 메타데이터 저장 (기존 형식 유지)
        manifest_path = os.path.join(DB_DIR, f"{index_name}_chunks.jsonl")
        with open(manifest_path, "w", encoding="utf-8") as fp:
            for ch in chunks:
                fp.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")

        logger.info(f"FAISS index and metadata saved for index '{index_name}' (LangChain compatible).")
        return True

    def process_and_save(self, pdf_path: str, index_name: str = "index", original_filename: str = None) -> bool:
        chunks = self.load_document(pdf_path, original_filename=original_filename)
        return self.save_to_faiss(chunks, index_name, original_filename)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="문서 로더 및 임베딩 생성기")
    parser.add_argument("pdf_path", help="처리할 PDF 파일 경로")
    parser.add_argument("--index-name", default="index", help="저장할 인덱스 이름")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="청크 크기")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help="청크 오버랩")
    args = parser.parse_args()

    loader = DocumentLoader(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    if loader.process_and_save(args.pdf_path, args.index_name):
        logger.info(f"SUCCESS: Document processed and saved: {args.pdf_path}")
        return 0
    else:
        logger.error(f"ERROR: Failed to process document: {args.pdf_path}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
