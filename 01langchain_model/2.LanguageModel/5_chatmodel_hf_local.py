# don;t run this code as it download the model locally in system

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from dotenv import load_dotenv
import os

load_dotenv()

print("HuggingFace API Key:", os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))

llm = HuggingFacePipeline.from_model_id(
    model_id="zai-org/GLM-4.5",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    pipeline_kwargs={
        temperature: 0.7,
        max_new_tokens: 100,
    }
)

model = ChatHuggingFace(llm=llm)

model_result = model.invoke("What is the capital of India?")
print(model_result)
print(model_result.content)  # Print the content of the response