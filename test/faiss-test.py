from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# 임베딩 모델 로드
embeddings = OpenAIEmbeddings()

# 텍스트 데이터
texts = [
    "랑체인은 LLM 애플리케이션 개발을 위한 프레임워크입니다.",
    "FAISS는 벡터 유사도 검색을 위한 라이브러리입니다.",
    "벡터스토어는 텍스트를 벡터로 변환하여 저장합니다.",
    "유사도 검색은 쿼리 벡터와 가장 유사한 벡터를 찾습니다."
]

try:
  print("faiss vectorstore found, loading...")
  # FAISS 벡터스토어 로드
  vectorstore = FAISS.load_local("db/faiss", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
except RuntimeError as e:
  print("faiss vectorstore not found, creating new one...")
  # FAISS 벡터스토어 생성
  vectorstore = FAISS.from_texts(texts, embeddings)
  # 벡터스토어 저장
  vectorstore.save_local("db/faiss")

# 유사도 검색
query = "FAISS의 역할은 무엇인가요?"
docs = vectorstore.similarity_search(query)

# 결과 출력
print(docs[0].page_content)
