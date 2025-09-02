from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

configurable_model = init_chat_model(
    temperature=0.2,
    model="gpt-5-mini",
    model_provider="openai"
)

prompt = ChatPromptTemplate.from_template(
  "{product}에 대해 한줄로 설명해줘"
)

output = StrOutputParser()

chain = prompt | configurable_model | output

print(chain.invoke({"product" : "아이스크림"}))
print(chain.invoke({"product" : "사과"}))