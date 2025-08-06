# claude model
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.7, max_completion_tokens=1000)

result = chat_model.invoke("What is the capital of France?")
print(result)
