from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm, max_completion_tokens=100)
 
# 1st prompt -> detail report
template1 = PromptTemplate(
    template='Write an detiled report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write an 5 line summary on given text. /n {text}',
    input_variables=['text']
)

########################################### without the use of StrOutputParser
# prompt1 = template1.invoke({'topic': 'black hole'})

# result1 = model.invoke(prompt1)

# prompt2 = template2.invoke({'text': result1.content})

# result = model.invoke(prompt2)

# print(result.content)


########################################### with the use of StrOutputParser
parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'black hole'})
print(result)

