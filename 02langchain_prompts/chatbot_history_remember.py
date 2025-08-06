# issue in this coe is that chat history is store in list but if chat history is long then it will not work
# so we need to use a different approach to store chat history
# we can use a dictionary to store chat history with user input as key and bot response as value
# This chatbot remembers past conversations and uses the chat history to provide context for responses.
# langchain solve this problem by messages

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct", 
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm, temperature=0.7, max_completion_tokens=100)

chat_history = []

while True:
    user_input = input("You: ")
    chat_history.append(user_input)
    if user_input.lower() == "exit":
        print("Exiting the chatbot. Goodbye!")
        break
    result = model.invoke(chat_history)
    chat_history.append(result.content)
    print("Bot:", result.content)

# Display chat history
print("\nChat History:", chat_history)