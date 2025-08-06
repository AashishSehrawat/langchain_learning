from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
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

chat_history = [
    SystemMessage(content="You are a helpful nutritionist."),
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        print("Exiting the chatbot. Goodbye!")
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("Bot:", result.content)

print("\nChat History:", chat_history)





# Example of using messages to maintain chat history

# message = [
#     SystemMessage(content="You are a helpful nutritionist."),
#     HumanMessage(content="What is the best diet for weight loss?"),
# ]

# result = model.invoke(message)

# message.append(AIMessage(content=result.content))

# print("Chat History:", message)