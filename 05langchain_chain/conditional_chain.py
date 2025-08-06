# hugging open source doesn't give certian output
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['Positive', 'Negative'] = Field(description="give the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template=("You are a strict JSON output generator.\n"
        "Classify the sentiment of the following feedback into Positive or Negative.\n"
        "Respond ONLY in the following JSON format and nothing else:\n"
        "{format}\n\n"
        "Feedback: {feedback}"),
    input_variables=['feedback'],
    partial_variables={'format': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="write an appropiate response to this positive feedback \n {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="write an appropiate response to this negative feedback \n {feedback}",
    input_variables=['feedback']
)

brach_chain = RunnableBranch(
    (lambda x: x.sentiment == 'Positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'Negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not found sentiment")
)

chain = classifier_chain | brach_chain

result = chain.invoke({'feedback': """This phone freezes every time I open more than two apps and the touchscreen often stops responding.
"""})
print(result)
