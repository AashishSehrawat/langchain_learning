# HuggingFaceEndpoint is used to using the huggingface by api

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

print("HuggingFace API Key:", os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))


# repo_id = which model you want to use
llm = HuggingFaceEndpoint(
    repo_id="zai-org/GLM-4.5",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm, temperature=0.7, max_completion_tokens=1000)

model_result = model.invoke("What is the capital of India?")
print(model_result)
print(model_result.content)  # Print the content of the response