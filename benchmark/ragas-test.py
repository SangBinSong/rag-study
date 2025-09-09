# uv run -m benchmark.ragas-test

from datetime import datetime
from datasets import Dataset
from ragas import evaluate
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

from dotenv import load_dotenv

load_dotenv()

import os
os.environ["LANGSMITH_PROJECT"] = "RAGAS-TESTSET"

# Arrow 포맷에서 불러오기
save_path_arrow = "./benchmark/rag_eval_dataset_arrow"
loaded_dataset = Dataset.load_from_disk(save_path_arrow)
print(f"'{save_path_arrow}' 경로에서 데이터셋을 불러왔습니다.")


llm = ChatOpenAI(model="gpt-5-nano")
evaluator_llm = LangchainLLMWrapper(llm,bypass_temperature=True)
print("\nRAG 평가 실행 중...")
# RAG 평가 실행
result = evaluate(
    dataset=loaded_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
    llm=evaluator_llm
)

result.to_pandas().to_csv(f"./benchmark/ragas-test-result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)

# 결과 출력
print("\nRAG 평가 결과:")
print(result)