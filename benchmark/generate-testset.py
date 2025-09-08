# https://docs.ragas.io/en/stable/getstarted/rag_testset_generation/
# uv run -m benchmark.generate-testset
from datetime import datetime
import logging
import os
import asyncio
from dotenv import load_dotenv
from ragas.run_config import RunConfig
from ragas.testset.transforms import HeadlineSplitter
from ragas.testset.transforms.extractors.llm_based import NERExtractor

# HTTP 요청 로그 숨기기
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
os.environ["OPENAI_LOG_LEVEL"] = "ERROR"

from module.vector_db import VectorDB

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)

load_dotenv()

async def main():
    vector_db = VectorDB(storage_path="./db/faiss")
    documents = list(vector_db.vectorstore.docstore._dict.values())

    llm_model = "gpt-4.1-nano"
    embedding_model = "text-embedding-3-small"

    generator_llm = LangchainLLMWrapper(ChatOpenAI(model=llm_model))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_model))

    personas = [
        # 페르소나 1: 국가 AI 정책을 총괄하는 고위 결정권자
        Persona(
            name="National AI Strategy Planning Officer",
            role_description="다른 국가들(영국, 미국, 호주)의 AI 전략을 비교 분석하여, 우리나라의 국가 AI 전략 방향성과 거버넌스 구축에 대한 인사이트를 얻고 싶어 합니다.",
        ),
        # 페르소나 2: 소속 부처에 AI를 도입해야 하는 실무 담당자
        Persona(
            name="AI Implementation Manager",
            role_description="소속 부처에 AI 기술을 도입하고 활용해야 하는 실무자로서, 구체적인 도입 절차, 활용 가이드라인, 그리고 AI 책임자(CAIO)의 역할과 같은 실질적인 정보가 필요합니다.",
        ),
        # 페르소나 3: 공공부문 AI 정책을 연구하는 연구원
        Persona(
            name="Public Administration Researcher",
            role_description="해외 주요국의 공공부문 AI 도입 전략의 특징과 차이점을 심도 있게 분석하여, 각 국가별 정책 접근 방식의 장단점을 파악하고 향후 정책 연구를 위한 기초 자료로 활용하고자 합니다.",
        )
    ]

    distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1.0),
    ]


    transforms = [HeadlineSplitter(), NERExtractor()]

    for query, _ in distribution:
        prompts = await query.adapt_prompts("korean", llm=generator_llm)
        query.set_prompts(**prompts)

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        persona_list=personas,
    )

    dataset = generator.generate_with_langchain_docs(
        documents[:],
        testset_size=30,
        transforms=transforms,
        query_distribution=distribution,
        with_debugging_logs=True,
        run_config=RunConfig(
            timeout=1000,
            max_workers=5,  # 병렬 처리 제한
            max_retries=10
        )
    )

    dataset.to_jsonl(f"./benchmark/testset_{llm_model.replace(':', '_')}_{embedding_model.replace(':', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")

if __name__ == "__main__":
    asyncio.run(main())