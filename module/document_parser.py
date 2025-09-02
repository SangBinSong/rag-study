"""
LangChain을 사용한 범용 문서 로더 모듈
PDF, TXT, DOCX, HTML, CSV 등 다양한 문서 형식을 지원합니다.
"""

import logging
from pathlib import Path
from typing import List, Union, Dict, Any
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    JSONLoader
)
from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from magika import Magika

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentLoader:
    """AI 기반 파일 타입 검출을 지원하는 다중 포맷 문서 로더"""
    
    # Magika 파일 타입과 LangChain 로더 매핑
    MAGIKA_LOADER_MAPPING = {
        'pdf': PyMuPDFLoader,
        'txt': TextLoader,
        'csv': CSVLoader,
        'html': UnstructuredHTMLLoader,
        'docx': UnstructuredWordDocumentLoader,
        'doc': UnstructuredWordDocumentLoader,
        'pptx': UnstructuredPowerPointLoader,
        'ppt': UnstructuredPowerPointLoader,
        'xlsx': UnstructuredExcelLoader,
        'xls': UnstructuredExcelLoader,
        'markdown': UnstructuredMarkdownLoader,
        'json': JSONLoader
    }
    
    # 확장자 기반 폴백 매핑 (Magika 검출 실패 시)
    EXTENSION_LOADER_MAPPING = {
        '.pdf': PyMuPDFLoader,
        '.txt': TextLoader,
        '.csv': CSVLoader,
        '.html': UnstructuredHTMLLoader,
        '.htm': UnstructuredHTMLLoader,
        '.docx': UnstructuredWordDocumentLoader,
        '.doc': UnstructuredWordDocumentLoader,
        '.pptx': UnstructuredPowerPointLoader,
        '.ppt': UnstructuredPowerPointLoader,
        '.xlsx': UnstructuredExcelLoader,
        '.xls': UnstructuredExcelLoader,
        '.md': UnstructuredMarkdownLoader,
        '.json': JSONLoader
    }
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        문서 로더 초기화
        
        Args:
            chunk_size: 텍스트 청크 분할 크기
            chunk_overlap: 청크 간 겹침 크기
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        # AI 기반 파일 타입 검출을 위한 Magika 초기화
        self.magika = Magika()
    
    def _detect_file_type(self, file_path: Path) -> str:
        """
        Magika AI를 사용한 파일 타입 검출
        
        Args:
            file_path: 파일 경로
            
        Returns:
            검출된 파일 타입 문자열
        """
        try:
            result = self.magika.identify_path(file_path)
            detected_type = result.output.ct_label
            logger.info(f"Magika detected file type: {detected_type} for {file_path}")
            return detected_type
        except Exception as e:
            logger.warning(f"Magika detection failed for {file_path}: {str(e)}")
            return None
    
    def load_document(
        self, 
        file_path: Union[str, Path],
        encoding: str = 'utf-8',
        **kwargs
    ) -> List[Document]:
        """
        파일 경로에서 단일 문서 로드
        
        Args:
            file_path: 문서 파일 경로
            encoding: 텍스트 인코딩 (기본값: utf-8)
            **kwargs: 특정 로더용 추가 인자
            
        Returns:
            Document 객체 리스트
            
        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우
            ValueError: 지원되지 않는 파일 형식인 경우
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Magika AI 기반 검출만 사용
        detected_type = self._detect_file_type(file_path)
        
        if not detected_type or detected_type not in self.MAGIKA_LOADER_MAPPING:
            raise ValueError(f"Magika가 지원하지 않는 파일 형식을 검출했습니다: {detected_type}")
        
        loader_class = self.MAGIKA_LOADER_MAPPING[detected_type]
        logger.info(f"Magika 검출 사용: {detected_type}")
        
        try:
            # 검출된 타입에 따른 특정 로더 요구사항 처리
            if detected_type == 'txt':
                loader = loader_class(str(file_path), encoding=encoding)
            elif detected_type == 'csv':
                loader = loader_class(str(file_path), encoding=encoding, **kwargs)
            elif detected_type == 'json':
                # JSON 로더는 jq_schema 매개변수가 필요
                jq_schema = kwargs.get('jq_schema', '.')
                loader = loader_class(str(file_path), jq_schema=jq_schema)
            else:
                loader = loader_class(str(file_path), **kwargs)
            
            documents = loader.load()
            
            # Magika 검출 결과를 포함한 메타데이터 추가
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path),
                    'detected_type': detected_type,
                    'file_size': file_path.stat().st_size,
                    'file_name': file_path.name,
                    'detection_method': 'magika'
                })
            
            logger.info(f"Successfully loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def load_directory(
        self,
        directory_path: Union[str, Path],
        glob_pattern: str = "*",
        recursive: bool = True,
        show_progress: bool = False,
        **kwargs
    ) -> List[Document]:
        """
        Load all supported documents from a directory
        
        Args:
            directory_path: Path to directory
            glob_pattern: Pattern to match files (default: "*")
            recursive: Whether to search subdirectories
            show_progress: Whether to show loading progress
            **kwargs: Additional arguments for loaders
            
        Returns:
            List of Document objects from all loaded files
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory_path}")
        
        all_documents = []
        supported_magika_types = list(self.MAGIKA_LOADER_MAPPING.keys())
        
        # Find all files (let Magika decide if they're supported)
        pattern = "**/*" if recursive else "*"
        files = [f for f in directory_path.glob(pattern) if f.is_file()]
        
        if not files:
            logger.warning(f"No supported files found in {directory_path}")
            return []
        
        logger.info(f"Found {len(files)} supported files in {directory_path}")
        
        processed_files = 0
        skipped_files = 0
        
        for file_path in files:
            if show_progress:
                print(f"Processing: {file_path.name}")
            
            try:
                # Check if file is supported by Magika detection
                detected_type = self._detect_file_type(file_path)
                
                if detected_type and detected_type in supported_magika_types:
                    documents = self.load_document(file_path, **kwargs)
                    all_documents.extend(documents)
                    processed_files += 1
                    if show_progress:
                        print(f"✅ Loaded: {file_path.name} ({detected_type})")
                else:
                    skipped_files += 1
                    if show_progress:
                        print(f"⏭️ Skipped: {file_path.name} (unsupported: {detected_type})")
                        
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {str(e)}")
                skipped_files += 1
                continue
        
        logger.info(f"Processing complete: {processed_files} loaded, {skipped_files} skipped, {len(all_documents)} total documents")
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of split Document objects
        """
        if not documents:
            return []
        
        split_docs = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")
        return split_docs
    
    def load_and_split(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> List[Document]:
        """
        Convenience method to load and split a document in one step
        
        Args:
            file_path: Path to the document file
            **kwargs: Additional arguments for loader
            
        Returns:
            List of split Document objects
        """
        documents = self.load_document(file_path, **kwargs)
        return self.split_documents(documents)
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """
        Get list of supported file formats
        
        Returns:
            Dictionary with magika types and file extensions
        """
        return {
            'magika_types': list(self.MAGIKA_LOADER_MAPPING.keys()),
            'extensions': list(self.EXTENSION_LOADER_MAPPING.keys())
        }
    
    def get_document_info(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get information about loaded documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary with document statistics
        """
        if not documents:
            return {}
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        file_types = {}
        detected_types = {}
        detection_methods = {'magika': 0, 'extension': 0}
        sources = set()
        
        for doc in documents:
            # Magika detection stats
            detected_type = doc.metadata.get('detected_type', 'unknown')
            detected_types[detected_type] = detected_types.get(detected_type, 0) + 1
            
            # Detection method stats  
            method = doc.metadata.get('detection_method', 'unknown')
            if method in detection_methods:
                detection_methods[method] += 1
                
            sources.add(doc.metadata.get('source', 'unknown'))
        
        return {
            'total_documents': len(documents),
            'total_characters': total_chars,
            'total_sources': len(sources),
            'detected_types': detected_types,
            'detection_methods': detection_methods,
            'average_doc_length': total_chars // len(documents) if documents else 0
        }


def load_documents(
    path: Union[str, Path],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    split_documents: bool = True,
    **kwargs
) -> List[Document]:
    """
    파일 또는 디렉터리에서 문서를 로드하는 편의 함수
    
    Args:
        path: 파일 또는 디렉터리 경로
        chunk_size: 텍스트 청크 분할 크기
        chunk_overlap: 청크 간 겹침 크기
        split_documents: 문서를 청크로 분할할지 여부
        **kwargs: 로더용 추가 인자
        
    Returns:
        Document 객체 리스트
    """
    loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    path = Path(path)
    
    if path.is_file():
        documents = loader.load_document(path, **kwargs)
    elif path.is_dir():
        documents = loader.load_directory(path, **kwargs)
    else:
        raise ValueError(f"Path does not exist: {path}")
    
    if split_documents:
        documents = loader.split_documents(documents)
    
    return documents


if __name__ == "__main__":
    # 실제 PDF 파일을 사용한 예제
    loader = DocumentLoader(chunk_size=1000, chunk_overlap=200)
    
    # 지원 형식 확인
    print("🔧 지원 형식:", loader.get_supported_formats())
    print()
    
    # 예제 1: Magika 검출을 사용한 단일 PDF 파일 로드
    try:
        pdf_path = "sample/국가별 공공부문 AI 도입 및 활용 전략.pdf"
        print(f"📄 PDF 로딩: {pdf_path}")
        documents = loader.load_and_split(pdf_path)
        
        print(f"✅ 성공적으로 {len(documents)}개 문서 청크를 로드했습니다")
        
        # 첫 번째 문서 정보 표시
        if documents:
            first_doc = documents[0]
            print(f"📊 첫 번째 청크 미리보기 (처음 200자):")
            print(f"   내용: {first_doc.page_content[:200]}...")
            print(f"   메타데이터: {first_doc.metadata}")
        print()
        
        # 문서 통계 확인
        info = loader.get_document_info(documents)
        print("📈 문서 통계:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        print()
        
    except Exception as e:
        print(f"❌ PDF 로딩 오류: {e}")
        print()
    
    # 예제 2: 샘플 디렉터리의 모든 문서 로드
    try:
        sample_dir = "sample"
        print(f"📁 디렉터리의 모든 문서 로딩: {sample_dir}")
        all_documents = loader.load_directory(sample_dir, show_progress=True, recursive=True)
        
        if all_documents:
            dir_info = loader.get_document_info(all_documents)
            print(f"📈 디렉터리 통계:")
            for key, value in dir_info.items():
                print(f"   {key}: {value}")
        else:
            print("   문서를 찾거나 로드하지 못했습니다")
            
    except Exception as e:
        print(f"❌ 디렉터리 로딩 오류: {e}")
    
    # 예제 3: 편의 함수 사용
    print("\n🚀 편의 함수 사용:")
    try:
        quick_docs = load_documents("sample/국가별 공공부문 AI 도입 및 활용 전략.pdf", 
                                  chunk_size=500, 
                                  split_documents=True)
        print(f"✅ 빠른 로드: {len(quick_docs)}개 청크")
    except Exception as e:
        print(f"❌ 빠른 로드 오류: {e}")