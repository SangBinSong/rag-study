# https://docs.ragas.io/en/stable/getstarted/rag_testset_generation/
# uv run -m banchmark.generate-testset
import datetime
import logging
import os
from dotenv import load_dotenv

# HTTP 요청 로그 숨기기
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
os.environ["OPENAI_LOG_LEVEL"] = "ERROR"

from module.vector_db import VectorDB

from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.testset import TestsetGenerator
import openai

load_dotenv()

vector_db = VectorDB(storage_path="./db/faiss")
documents = list(vector_db.vectorstore.docstore._dict.values())

model = "gpt-4.1-nano"

generator_llm = LangchainLLMWrapper(ChatOpenAI(model=model))
openai_client = openai.OpenAI();
generator_embeddings = OpenAIEmbeddings(client=openai_client, model="text-embedding-3-small")
generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
generator.adapt

dataset = generator.generate_with_langchain_docs(documents, testset_size=1, with_debugging_logs=True)

dataset.to_csv(f"testset_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
