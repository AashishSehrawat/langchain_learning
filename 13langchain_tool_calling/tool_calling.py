from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------- create an tool
@tool
def multiply(a:int, b:int) -> int:
    """Given 2 numbers a and b this tool returns their product"""
    return a*b



# ----------------------------------- binding the tool
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
llm_with_tools = llm.bind_tools([multiply])



# ----------------------------------- tool calling
# llm return an tool_calls=[{'name': 'multiply', 'args': {'b': 5.0, 'a': 3.0}, 'id': '976549d2-3a0c-4669-9c78-3ddfcf1d45f2', 'type': 'tool_call'}]. It doesn't call the return answer as llm conn't call the tool. llm only suggest the tool based on query

query = HumanMessage("can you multiply 3 with 5")
message = [query]
result = llm_with_tools.invoke(message)
message.append(result)
result_args = result.tool_calls[0]



# ------------------------------------ tool exceution
tool_message = multiply.invoke(result_args)
message.append(tool_message)

# print(message)
final_result =  llm_with_tools.invoke(message)
print(final_result)






