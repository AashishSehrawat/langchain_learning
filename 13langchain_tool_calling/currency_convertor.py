from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage
from typing import Annotated
from dotenv import load_dotenv
import requests
import json

load_dotenv()

# ---------------------------------- tool created
@tool 
def get_currency_factor(base_currency: str, target_currency: str) -> float:
    """
    This function fetches the currency conversion factor between a given base currency and a target currency
    """
    url = f"https://v6.exchangerate-api.com/v6/e7926491f0f6f3e391cc9d44/pair/{base_currency}/{target_currency}"

    response = requests.get(url)
    return response.json()

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """
    given a currency conversion rate this function calculates the target currency value from a given base currency value
    """
    return base_currency_value*conversion_rate


    
# ---------------------------------------- tool bind
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
llm_with_tools = llm.bind_tools([get_currency_factor, convert])



# ----------------------------------------- tool calling
message = [HumanMessage('What is the conversion factor between USD and INR, and based on that can you convert 10 usd to inr')]

ai_message = llm_with_tools.invoke(message)
message.append(ai_message)

for tool_call in ai_message.tool_calls: 
    # execute the 1st tool and get the value of conversion rate
    if tool_call['name'] == "get_currency_factor":
        tool_message1 = get_currency_factor.invoke(tool_call)
        
        # fetch the conversation_rate
        data = json.loads(tool_message1.content)

        # Extract the conversion_rate field
        converation_rate = data["conversion_rate"]
        message.append(tool_message1)
        
    # execute the 2nd tool using the conversion rate from tool 1
    if tool_call['name'] == "convert":
        tool_call['args']['conversion_rate'] = converation_rate
        tool_message2 = convert.invoke(tool_call)
        message.append(tool_message2)

print(llm_with_tools.invoke(message))
        