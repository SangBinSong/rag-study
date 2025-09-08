# uv run -m benchmark.rag-test-data
from datasets import load_dataset
from module.vector_db import VectorDB
from module.simple_rag_chat import SimpleRAGChat
from langchain_openai import ChatOpenAI

vector_db = VectorDB(storage_path="./db/faiss")
rag = SimpleRAGChat(
    llm=ChatOpenAI(model="gpt-4.1-mini", temperature=0.1),
    vector_store=vector_db,
    config=SimpleRAGChat.RAGConfig(
        use_history_prompt=False,
        use_history_query=False
    )
)

test_dataset = load_dataset("json", data_files="./benchmark/testset_gpt-4.1-nano_text-embedding-3-small_20250909_044039.jsonl", split="train")

# testset의 각 질문에 대해 RAG 응답 생성
print("RAG 응답 생성 중...")
responses = []
retrieved_contexts = []

for i, row in enumerate(test_dataset):
    question = row['user_input']
    print(f"질문 {i+1}/{len(test_dataset)}: {question[:50]}...")
    
    # RAG 시스템으로 응답 생성
    response, documents = rag.send_with_documents(question)
    responses.append(response)
    retrieved_contexts.append([doc.page_content for doc in documents])

# testset에 응답과 컨텍스트 추가
updated_dataset = test_dataset.add_column('response', responses).add_column('retrieved_contexts', retrieved_contexts)

save_path_arrow = "./benchmark/rag_eval_dataset_arrow"
updated_dataset.save_to_disk(save_path_arrow)
print(f"데이터셋을 Arrow 포맷으로 '{save_path_arrow}' 경로에 저장했습니다.")

