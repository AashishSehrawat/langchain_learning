# This chatbot we created doesn't remember past conversations.
# It only responds to the current input without any context from previous messages.

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="zai-org/GLM-4.5",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm, temperature=0.7, max_completion_tokens=100)

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Exiting the chatbot. Goodbye!")
        break
    result = model.invoke(user_input)
    print("Bot:", result.content)