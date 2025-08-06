# This code doesns't need to be run, it's just an example of how to use the OpenAI API with LangChain.

# using llm without langchain
# all llm inheritance from BaseLLM class
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo-instruct")

result = llm.invoke("What is the capital of France?")

print(result)
