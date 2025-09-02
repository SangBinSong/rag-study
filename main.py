import os
from pathlib import Path
from module.document_parser import load_documents
from module.vector_db import VectorDB


def demo_document_loading():
    """Demonstrate Magika-based document loading"""
    print("🔍 RAG Study Demo - AI 기반 문서 로딩 + 벡터 검색")
    print("=" * 60)
    
    # Get file path from user
    print("📁 파일 선택 옵션:")
    print("1. 기본 샘플 파일 사용")
    print("2. 직접 파일 경로 입력")
    
    choice = input("\n선택하세요 (1 또는 2): ").strip()
    
    if choice == "2":
        file_path = input("📄 파일 경로를 입력하세요: ").strip()
        # Remove quotes if present
        file_path = file_path.strip('"').strip("'")
    else:
        file_path = "sample/국가별 공공부문 AI 도입 및 활용 전략.pdf"
    
    if not Path(file_path).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return None, None
    
    print(f"\n📄 문서 로딩: {file_path}")
    try:
        documents = load_documents(file_path, chunk_size=1000, chunk_overlap=200, split_documents=True)
        print(f"✅ 성공! {len(documents)}개 청크로 분할됨")
        
        # Show document info
        if documents:
            total_chars = sum(len(doc.page_content) for doc in documents)
            detected_types = {}
            detection_methods = {'magika': 0}
            
            for doc in documents:
                detected_type = doc.metadata.get('detected_type', 'unknown')
                detected_types[detected_type] = detected_types.get(detected_type, 0) + 1
                detection_methods['magika'] += 1
            
            print(f"📊 문서 통계:")
            print(f"   - 총 문자 수: {total_chars:,}")
            print(f"   - 평균 청크 길이: {total_chars // len(documents):,}")
            print(f"   - 검출된 파일 타입: {detected_types}")
            print(f"   - 검출 방식: {detection_methods}")
            print()
        
        return documents, None
        
    except Exception as e:
        print(f"❌ 문서 로딩 실패: {e}")
        return None, None


def demo_vector_database(documents):
    """Demonstrate vector database operations"""
    if not documents:
        print("⚠️ 문서가 없어 벡터 데이터베이스 데모를 건너뜁니다.")
        return
    
    print("🗂️ 벡터 데이터베이스 데모")
    print("=" * 30)
    
    # Initialize vector database
    db_path = "./db/rag_demo"
    vector_db = VectorDB(storage_path=db_path)
    
    print(f"📈 초기 상태: {vector_db.get_stats()}")
    
    # Add documents if empty
    if vector_db.is_empty():
        # Ask user if they want to add documents
        add_choice = input("\n📝 문서를 벡터 데이터베이스에 추가하시겠습니까? (y/n): ").strip().lower()
        if add_choice in ['y', 'yes', '예', 'ㅇ']:
            print("📝 벡터 데이터베이스에 문서 추가 중...")
            vector_db.add_documents(documents[:10])  # First 10 chunks for demo
            print("✅ 문서 추가 완료!")
        else:
            print("ℹ️ 문서 추가를 건너뛰었습니다.")
            return
    else:
        print("📚 기존 데이터베이스 사용 중")
    
    print(f"📈 현재 상태: {vector_db.get_stats()}")
    print()
    
    # Perform similarity searches
    queries = [
        "AI 도입 전략은 무엇인가?",
        "공공부문에서의 인공지능 활용",
        "디지털 전환과 혁신"
    ]
    
    print("🔍 유사도 검색 데모:")
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. 질문: {query}")
        try:
            results = vector_db.similarity_search(query, k=2)
            for j, doc in enumerate(results, 1):
                content_preview = doc.page_content[:100].replace('\n', ' ')
                print(f"   답변 {j}: {content_preview}...")
                if doc.metadata.get('detected_type'):
                    print(f"   📄 파일 타입: {doc.metadata['detected_type']}")
        except Exception as e:
            print(f"   ❌ 검색 실패: {e}")
    
    # Ask user if they want to save to disk
    save_choice = input("\n💾 벡터 데이터베이스를 디스크에 저장하시겠습니까? (y/n): ").strip().lower()
    if save_choice in ['y', 'yes', '예', 'ㅇ']:
        vector_db.save()
        print("✅ 벡터 데이터베이스가 디스크에 저장되었습니다!")
    else:
        print("ℹ️ 메모리에만 유지됩니다 (프로그램 종료 시 삭제)")


def main():
    """Main demo function"""
    print("🚀 RAG Study 프로젝트 데모")
    print("Magika AI 기반 문서 로딩 + FAISS 벡터 검색\n")
    
    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("export OPENAI_API_KEY='your-key-here'")
        print("계속해서 데모를 실행하지만 벡터 검색은 실패할 수 있습니다.\n")
    
    # Demo 1: Document Loading
    documents, _ = demo_document_loading()
    
    # Demo 2: Vector Database
    if documents:
        print()
        demo_vector_database(documents)
    
    print("\n🎉 데모 완료!")
    print("\n💡 추가 테스트:")
    print("  uv run module/document-load.py  # 문서 로더 단독 테스트")
    print("  uv run module/vector-db.py      # 벡터 DB 단독 테스트")


if __name__ == "__main__":
    main()
