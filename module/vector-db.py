import logging
import os
import json
from typing import List, Optional, Dict, Any

import faiss
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from pydantic.dataclasses import dataclass

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDB:
    """
    FAISS 기반 벡터 데이터베이스 래퍼
    
    메모리 중심 접근법:
    - FAISS는 본질적으로 인메모리 기반
    - 선택적 영속성 (재임베딩 방지용)
    - 단순하고 명확한 API
    """

    @dataclass
    class EmbeddingInfo:
        model_class: str
        model_name: str
        embedding_size: int
    
    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        storage_path: Optional[str] = None,
        override: bool = True
    ):
        """
        VectorDB 초기화
        
        Args:
            embeddings: 임베딩 모델. None이면 OpenAI 기본값
            storage_path: 저장 경로. None이면 순수 메모리 모드
            override: 임베딩 불일치 시 덮어쓰기 여부
        """
        self._storage_path = storage_path
        self._override = override
        self.embeddings = embeddings or OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore: Optional[FAISS] = None

        self._embedding_info = self.EmbeddingInfo(
            model_class=self.embeddings.__class__.__name__,
            model_name=getattr(self.embeddings, 'model', 'unknown'),
            embedding_size=len(self.embeddings.embed_query("test"))
        )
        
        # storage_path가 있으면 초기화 로직 수행
        if self._storage_path:
            self._initialize_vectorstore()
        else:
            self._create_empty_vectorstore()
            
    def _get_embedding_info_path(self) -> Optional[str]:
        """임베딩 정보 파일 경로 반환"""
        if not self._storage_path:
            return None
        return f"{self._storage_path}/embedding_info.json"
    
    def _save_embedding_info(self) -> None:
        """임베딩 모델 정보 저장"""
        info_path = self._get_embedding_info_path()
        if not info_path:
            return
        
        embedding_info = self._embedding_info.model_dump_json()

        try:
            with open(info_path, 'w') as f:
                f.write(embedding_info)
            logger.info(f"임베딩 정보 저장: {info_path}")
        except Exception as e:
            logger.warning(f"임베딩 정보 저장 실패: {e}")
    
    def _load_and_check_embedding_info(self) -> bool:
        """임베딩 정보 로드하고 현재 모델과 비교"""
        info_path = self._get_embedding_info_path()
        if not info_path or not os.path.exists(info_path):
            logger.warning("임베딩 정보 파일이 없습니다. 호환성을 확인할 수 없습니다.")
            return True  # 정보가 없어도 시도는 해봄
            
        try:
            with open(info_path, 'r') as f:
                saved_info = json.load(f)
            
            # JSON 문자열이 아닌 딕셔너리이므로 model_validate 사용
            saved_info = self.EmbeddingInfo.model_validate(saved_info)
            current_info = self._embedding_info

            # 임베딩 크기 비교 (가장 중요)
            if saved_info.embedding_size != current_info.embedding_size:
                logger.error(f"임베딩 크기 불일치: 저장={saved_info.embedding_size}, 현재={current_info.embedding_size}")
                return False
                
            # 모델 정보 경고
            if saved_info.model_class != current_info.model_class and self._override:
                logger.warning(f"임베딩 모델 클래스 다름: 저장={saved_info.model_class}, 현재={current_info.model_class}")
                
            if saved_info.model_name != current_info.model_name:
                if self._override:
                    logger.warning(f"임베딩 모델명 다름: 저장={saved_info.model_name}, 현재={current_info.model_name}")
                else:
                    logger.error(f"임베딩 모델명 불일치: 저장된 크기={saved_info.embedding_size}, 현재 크기={current_info.embedding_size}")
                    return False
            
            logger.info(f"임베딩 정보 확인 완료")
            return True
            
        except Exception as e:
            logger.warning(f"임베딩 정보 확인 실패: {e}")
            return True  # 확인 실패해도 시도는 해봄
    
    def _initialize_vectorstore(self) -> None:
        """벡터스토어 초기화 로직"""
        embedding_info_exists = self._get_embedding_info_path() and os.path.exists(self._get_embedding_info_path())
        faiss_exists = os.path.exists(self._storage_path)
        
        logger.info(f"초기화 상태 - 임베딩 정보: {embedding_info_exists}, FAISS: {faiss_exists}")
        
        if not embedding_info_exists and not faiss_exists:
            # 2.3: 임베딩 파일도 없고 faiss도 없는 경우 - 빈 faiss 생성만
            logger.info("새로운 벡터스토어 초기화: 빈 FAISS 생성")
            self._create_empty_vectorstore()
                
        elif not embedding_info_exists and faiss_exists:
            # 2.1: 임베딩 파일이 없는데 faiss가 있는 경우
            if self._override:
                logger.info("덮어쓰기 모드: 현재 임베딩으로 FAISS 로드")
                try:
                    self.vectorstore = FAISS.load_local(
                        self._storage_path, 
                        self.embeddings, 
                        allow_dangerous_deserialization=True
                    )
                    logger.info("기존 FAISS 로드 완료")
                except Exception as e:
                    logger.error(f"FAISS 로드 실패: {e}")
                    raise RuntimeError(f"FAISS 파일 로드 실패: {e}")
            else:
                raise ValueError(
                    f"임베딩 정보 파일이 없지만 FAISS 파일이 존재합니다: {self._storage_path}\n"
                    "override=True로 설정하거나 기존 파일을 삭제하세요."
                )
                
        elif embedding_info_exists and not faiss_exists:
            # 2.2: 임베딩 파일이 있는데 faiss가 없는 경우 - 빈 faiss 생성
            if self._load_and_check_embedding_info():
                logger.info("임베딩 정보 확인 완료: 빈 FAISS 생성")
                self._create_empty_vectorstore()
            else:
                raise ValueError(
                    f"임베딩 정보 파일의 모델이 현재 모델과 호환되지 않습니다.\n"
                    "다른 임베딩 모델을 사용하거나 기존 파일을 삭제하세요."
                )
                
        else:
            # 둘 다 있는 경우 - 기존 로직
            logger.info("기존 벡터스토어 로드 시도")
            self._load_existing_vectorstore()
    
    def _create_empty_vectorstore(self) -> None:
        """빈 벡터스토어 생성 (private 메서드)"""
        try:
            
            index = faiss.IndexFlatL2(self._embedding_info.embedding_size)
            self.vectorstore = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )
            logger.info("빈 벡터스토어 생성 완료")
        except Exception as e:
            logger.error(f"빈 벡터스토어 생성 실패: {e}")
            self.vectorstore = None
    
    
    def _load_existing_vectorstore(self) -> None:
        """기존 벡터스토어 로드 (private 메서드)"""
        # 임베딩 정보 확인
        if not self._load_and_check_embedding_info():
            error_msg = f"임베딩 모델 호환성 불일치. 기존 파일: {self._storage_path}을 수동으로 삭제하고 다시 시도하세요."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        try:
            logger.info(f"기존 벡터스토어 로드 중: {self._storage_path}")
            self.vectorstore = FAISS.load_local(
                self._storage_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            logger.info("기존 벡터스토어 로드 완료")
        except Exception as e:
            error_msg = f"벡터스토어 파일이 손상되었습니다. 파일: {self._storage_path}을 수동으로 삭제하고 다시 시도하세요. 원인: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> List[str]:
        """텍스트 추가"""
        if not self.vectorstore:
            raise RuntimeError("벡터스토어가 초기화되지 않았습니다")
        
        return self.vectorstore.add_texts(texts, metadatas=metadatas)
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Document 객체 추가"""
        if not self.vectorstore:
            raise RuntimeError("벡터스토어가 초기화되지 않았습니다")
        
        return self.vectorstore.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 4, filter: Optional[dict] = None) -> List[Document]:
        """유사도 검색"""
        if not self.vectorstore:
            raise RuntimeError("벡터스토어가 초기화되지 않았습니다")
        
        return self.vectorstore.similarity_search(query, k=k, filter=filter)
    
    def save(self, storage_path: Optional[str] = None) -> None:
        """디스크에 저장 (선택적 영속성)"""
        if not self.vectorstore:
            raise RuntimeError("벡터스토어가 초기화되지 않았습니다")
        if not self._storage_path and not storage_path:
            raise RuntimeError("storage_path가 설정되지 않았습니다")

        if storage_path:
            self._storage_path = storage_path
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(self._storage_path), exist_ok=True)
        
        # 벡터스토어 저장
        self.vectorstore.save_local(self._storage_path)
        logger.info(f"벡터스토어 저장: {self._storage_path}")
        
        # 임베딩 정보 저장
        self._save_embedding_info()
    
    def as_retriever(self, **kwargs):
        """LangChain 리트리버로 변환"""
        if not self.vectorstore:
            raise RuntimeError("벡터스토어가 초기화되지 않았습니다")
        return self.vectorstore.as_retriever(**kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보"""
        if not self.vectorstore:
            return {
                "status": "not_initialized",
                "has_persistence": self._storage_path is not None
            }
        
        try:
            index_size = getattr(self.vectorstore.index, 'ntotal', 0)
            docstore_size = len(getattr(self.vectorstore.docstore, '_dict', {}))
            
            return {
                "status": "initialized",
                "index_size": index_size,
                "docstore_size": docstore_size,
                "has_persistence": self._storage_path is not None,
                "storage_path": self._storage_path
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# 사용 예제
def example_usage():
    """사용 예제"""
    print("=== VectorDB 사용 예제 ===")
    
    # 1. 메모리 모드 (빈 벡터스토어)
    print("\n1. 메모리 모드:")
    vector_db = VectorDB()  # 자동으로 빈 벡터스토어 생성
    
    if vector_db.vectorstore is None:
        print("메모리 모드: 벡터스토어가 생성되지 않음 (storage_path 없음)")
    
    # 2. 새로운 영속성 모드 (자동 초기화)
    print("\n2. 새로운 벡터스토어:")
    vector_db_new = VectorDB(storage_path="db/faiss_new")  # 자동으로 빈 벡터스토어 생성
    
    texts = [
        "랑체인은 LLM 애플리케이션 개발을 위한 프레임워크입니다.",
        "FAISS는 벡터 유사도 검색을 위한 라이브러리입니다.",
        "벡터스토어는 텍스트를 벡터로 변환하여 저장합니다."
    ]
    
    vector_db_new.add_texts(texts)
    
    # Document 객체로도 추가 가능
    documents = [
        Document(page_content="Document 객체 테스트", metadata={"source": "test"}),
        Document(page_content="메타데이터가 있는 문서", metadata={"type": "example"})
    ]
    vector_db_new.add_documents(documents)
    
    vector_db_new.save()
    results = vector_db_new.similarity_search("FAISS란 무엇인가요?", k=1)
    print(f"검색 결과: {results[0].page_content}")
    print(f"통계: {vector_db_new.get_stats()}")
    
    # 3. 기존 벡터스토어 로드 (자동)
    print("\n3. 기존 벡터스토어 로드:")
    vector_db_load = VectorDB(storage_path="db/faiss_new")  # 자동으로 기존 파일 로드
    results = vector_db_load.similarity_search("프레임워크", k=1)
    print(f"검색 결과: {results[0].page_content}")
    
    # 4. 덮어쓰기 모드
    print("\n4. 덮어쓰기 모드 (임베딩 불일치 시):")
    try:
        VectorDB(storage_path="db/faiss_new", override=True)
        print("덮어쓰기 모드: 성공적으로 로드됨")
    except Exception as e:
        print(f"덮어쓰기 모드 에러: {e}")


if __name__ == "__main__":
    example_usage()