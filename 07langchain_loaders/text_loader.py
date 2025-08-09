from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os 
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="explain the defination in following text: \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

loader = TextLoader("d:/langchain/07langchain_loaders/theory.txt", encoding='utf-8');
docs = loader.load()

# print(type(docs))  # list
# print(docs)
# print(docs[0].page_content)
# print(docs[0].metadata)

chain = prompt | model | parser;
result = chain.invoke({'text': docs[0].page_content})
print(result)

