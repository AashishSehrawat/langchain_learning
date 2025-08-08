# can convert any python function into runnable

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import os 

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="write a joke about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="explain the following joke \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

def word_counter(text):
    return len(text.split())

joke_chain = RunnableSequence(prompt1, model, parser)
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    # 'length': RunnableLambda(word_counter)
    'length': RunnableLambda(lambda x: len(x.split()))
})

chain = RunnableSequence(joke_chain, parallel_chain)
result = chain.invoke({'topic': 'cricket'})
print(result)
