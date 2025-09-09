# https://docs.ragas.io/en/stable/getstarted/rag_testset_generation/
# uv run -m benchmark.generate-testset
from datetime import datetime
import logging
import os
import asyncio
from dotenv import load_dotenv
from ragas.run_config import RunConfig
from ragas.testset.transforms.extractors.llm_based import NERExtractor, KeyphrasesExtractor

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

os.environ["LANGSMITH_PROJECT"] = "RAGAS-TESTSET"

async def main():
    vector_db = VectorDB(storage_path="./db/faiss")
    documents = list(vector_db.vectorstore.docstore._dict.values())

    llm_model = "gpt-4.1-mini"
    embedding_model = "text-embedding-3-small"

    generator_llm = LangchainLLMWrapper(ChatOpenAI(model=llm_model, temperature=0.1))
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

    # 커스텀 context 검색을 위한 synthesizer 설정
    synthesizer = SingleHopSpecificQuerySynthesizer(llm=generator_llm)
    
    # 더 많은 컨텍스트를 활용하기 위한 수정
    if hasattr(synthesizer, 'context_model') and hasattr(synthesizer.context_model, 'top_k'):
        synthesizer.context_model.top_k = 15
    
    distribution = [
        (synthesizer, 1.0),
    ]

    transforms = [NERExtractor(), KeyphrasesExtractor(llm=generator_llm)]

    for query, _ in distribution:
        prompts = await query.adapt_prompts("korean", llm=generator_llm)
        query.set_prompts(**prompts)

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        persona_list=personas,
    )
    
    # 더 많은 컨텍스트를 검색하도록 설정
    for synthesizer, _ in distribution:
        if hasattr(synthesizer, 'context_model'):
            synthesizer.context_model.top_k = 15  # 더 많은 컨텍스트 검색

    dataset = generator.generate_with_langchain_docs(
        documents[:],
        testset_size=30,  # 테스트용으로 줄임
        transforms=transforms,
        query_distribution=distribution,
        with_debugging_logs=True,
        run_config=RunConfig(
            timeout=1500,  # 시간 여유 증가
            max_workers=3,  # 병렬 처리 제한을 줄여서 안정성 증가
            max_retries=10
        )
    )

    filename = f"./benchmark/testset_{llm_model.replace(':', '_')}_{embedding_model.replace(':', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    
    dataset.to_jsonl(filename)
    
    print(f"테스트셋 생성 완료: {filename}")
    print(f"생성된 샘플 수: {len(dataset)}")

if __name__ == "__main__":
    asyncio.run(main())