# passing dynamic multiple inputs to model and these input is SystemMessage, HumanMessage, AIMessage specific 

from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm, temperature=0.7, max_completion_tokens=100)

chat_tempplate = ChatPromptTemplate([
    ('system', "You are a helpful {domain} expert."),
    ('human', "Explain in simple terms what is {topic}?"),
])

prompt = chat_tempplate.invoke({
    'domain': "cricket",
    'topic': "batting"
})

print("Prompt:", prompt)
