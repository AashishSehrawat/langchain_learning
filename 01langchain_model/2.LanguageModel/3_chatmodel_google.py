from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7, max_completion_tokens=1000)

result = chat_model.invoke("What is the capital of india?")
print(result)
print(result.content)  