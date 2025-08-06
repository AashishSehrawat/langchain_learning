from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from typing import TypedDict, Annotated
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="chat-completion",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm, temperature=0.7, max_completion_tokens=100)

# result = model.invoke("tell me about virat kholi")
# print(result.content)

# schema 
class Review(TypedDict):
    summary: Annotated[str, "A breif summary of review"]
    sentiment: Annotated[str, "Return the sentiment of the review"]

structured_model = model.with_structured_output(Review)    

result = structured_model.invoke("""the hardware is great, but the software feels bloated. There are too many pre-installed apps that can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.""")

print(result)
