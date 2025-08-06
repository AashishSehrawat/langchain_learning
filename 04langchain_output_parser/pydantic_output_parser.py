from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm, max_completion_tokens=100)

class Person(BaseModel):
    name: str = Field(description="name of the person")
    age: int = Field(ge=18, description="age of the person")
    city: str = Field(description="name of the city the person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age and city of an frictional {place} person \n {format}",
    input_variables=['place'],
    partial_variables={'format': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'place': 'indian'})
print(result)

