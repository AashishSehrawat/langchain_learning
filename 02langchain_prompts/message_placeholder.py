from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

# chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are an customer support agent'),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', '{query}')
])

chat_history = []
# load chat_history
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

query = input("Enter your querry: ")

# create prompt
prompt = chat_template.invoke({'chat_history': chat_history, 'query': HumanMessage(content=query)})

print(prompt)


