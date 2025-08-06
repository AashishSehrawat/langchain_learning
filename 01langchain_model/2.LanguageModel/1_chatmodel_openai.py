# all chat models inherit from BaseChatModel class

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


# temperature = 0.0 - 0.5 => accuracy
# temperature = 1.0 - 2.0 => creativity
# max_completion_tokens = general way to understand the length of the response in words
# max_completion_tokens = 1000 => 1000 words
chat_model = ChatOpenAI(model="gpt-4", temperature=0.7, max_completion_tokens=1000)

result = chat_model.invoke("What is the capital of France?")
print(result)



