from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os 

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="openai/gpt-oss-120b",
#     task="text-generation",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
# )

# model1 = ChatHuggingFace(llm=llm)
model2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", max_tokens=200)

prompt1 = PromptTemplate(
    template="generate a tweet about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='generate a linkdin post about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model2, parser),
    'linkdin': RunnableSequence(prompt2, model2, parser)
})

result = chain.invoke({'topic': "Generative ai"})
print(result)
print(result['tweet'])
print(result['linkdin'])
