# used in creating the conditional chains

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableBranch, RunnablePassthrough
from dotenv import load_dotenv
import os 

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='create an report on {topic}',
    input_variables=['topic']
)

prompt1 = PromptTemplate(
    template="summarize the following text \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

report_chain = RunnableSequence(prompt, model, parser)

branc_chain = RunnableBranch(
    (lambda x: len(x.split()) > 200, RunnableSequence(prompt1, model, parser)),
    RunnablePassthrough()
)

chain = RunnableSequence(report_chain, branc_chain)
result = chain.invoke({'topic': 'cricket'})
print(result)



