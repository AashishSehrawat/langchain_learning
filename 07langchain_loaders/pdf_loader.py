from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os 
from dotenv import load_dotenv

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     task="text-generation",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
# )

# model = ChatHuggingFace(llm=llm)

# prompt = PromptTemplate(
#     template="explain the defination in following text: \n {text}",
#     input_variables=['text']
# )

# parser = StrOutputParser()

loader = PyPDFLoader("d:/langchain/07langchain_loaders/genai.pdf")

docs = loader.load()
print(docs)
print(len(docs))
