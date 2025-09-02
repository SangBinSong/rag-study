"""
Universal Document Loader Module using LangChain
Supports various document formats including PDF, TXT, DOCX, HTML, CSV, etc.
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
    """Universal document loader supporting multiple file formats with AI-based file type detection"""
    
    # Magika file types to LangChain loader mapping
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
    
    # Fallback extension mapping (for when magika detection fails)
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
        Initialize DocumentLoader
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        # Initialize Magika for AI-based file type detection
        self.magika = Magika()
    
    def _detect_file_type(self, file_path: Path) -> str:
        """
        Detect file type using Magika AI-based detection
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected file type string
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
        Load a single document from file path
        
        Args:
            file_path: Path to the document file
            encoding: Text encoding (default: utf-8)
            **kwargs: Additional arguments for specific loaders
            
        Returns:
            List of Document objects
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Use Magika AI-based detection only
        detected_type = self._detect_file_type(file_path)
        
        if not detected_type or detected_type not in self.MAGIKA_LOADER_MAPPING:
            raise ValueError(f"Unsupported file format detected by Magika: {detected_type}")
        
        loader_class = self.MAGIKA_LOADER_MAPPING[detected_type]
        logger.info(f"Using Magika detection: {detected_type}")
        
        try:
            # Handle specific loader requirements based on detected type
            if detected_type == 'txt':
                loader = loader_class(str(file_path), encoding=encoding)
            elif detected_type == 'csv':
                loader = loader_class(str(file_path), encoding=encoding, **kwargs)
            elif detected_type == 'json':
                # JSON loader requires jq_schema parameter
                jq_schema = kwargs.get('jq_schema', '.')
                loader = loader_class(str(file_path), jq_schema=jq_schema)
            else:
                loader = loader_class(str(file_path), **kwargs)
            
            documents = loader.load()
            
            # Add metadata with Magika detection results
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
    Convenience function to load documents from a file or directory
    
    Args:
        path: Path to file or directory
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks  
        split_documents: Whether to split documents into chunks
        **kwargs: Additional arguments for loaders
        
    Returns:
        List of Document objects
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
    # Example usage with real PDF file
    loader = DocumentLoader(chunk_size=1000, chunk_overlap=200)
    
    # Get supported formats
    print("🔧 Supported formats:", loader.get_supported_formats())
    print()
    
    # Example 1: Load a single PDF file with Magika detection
    try:
        pdf_path = "sample/국가별 공공부문 AI 도입 및 활용 전략.pdf"
        print(f"📄 Loading PDF: {pdf_path}")
        documents = loader.load_and_split(pdf_path)
        
        print(f"✅ Successfully loaded {len(documents)} document chunks")
        
        # Show first document info
        if documents:
            first_doc = documents[0]
            print(f"📊 First chunk preview (first 200 chars):")
            print(f"   Content: {first_doc.page_content[:200]}...")
            print(f"   Metadata: {first_doc.metadata}")
        print()
        
        # Get document statistics
        info = loader.get_document_info(documents)
        print("📈 Document Statistics:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        print()
        
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        print()
    
    # Example 2: Load all documents from sample directory  
    try:
        sample_dir = "sample"
        print(f"📁 Loading all documents from: {sample_dir}")
        all_documents = loader.load_directory(sample_dir, show_progress=True, recursive=True)
        
        if all_documents:
            dir_info = loader.get_document_info(all_documents)
            print(f"📈 Directory Statistics:")
            for key, value in dir_info.items():
                print(f"   {key}: {value}")
        else:
            print("   No documents found or loaded")
            
    except Exception as e:
        print(f"❌ Error loading directory: {e}")
    
    # Example 3: Using convenience function
    print("\n🚀 Using convenience function:")
    try:
        quick_docs = load_documents("sample/국가별 공공부문 AI 도입 및 활용 전략.pdf", 
                                  chunk_size=500, 
                                  split_documents=True)
        print(f"✅ Quick load: {len(quick_docs)} chunks")
    except Exception as e:
        print(f"❌ Quick load error: {e}")