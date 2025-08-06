from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm, max_completion_tokens=100)

parser = JsonOutputParser()

template = PromptTemplate(
    template="give me a name, age and city of an frictional character \n {format}",
    input_variables=[],
    partial_variables={'format': parser.get_format_instructions()}
)

###################################### option 1
# prompt = template.format()
# print(prompt)

# result = model.invoke(prompt)
# print("result: ", result)

# final_result = parser.parse(result.content)
# print("final result: ", final_result)

################################### option 2
chain = template | model | parser
result = chain.invoke({})

print(result)
