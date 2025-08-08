# its doesn't perform any operation on input. So, the output passed through this runnable is same as input

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
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

joke_generator_chain = RunnableSequence(prompt1, model, parser)
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_generator_chain, parallel_chain)
result = final_chain.invoke({'topic': 'cricket'})
print(result)




