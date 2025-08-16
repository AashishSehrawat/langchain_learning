# ------------------------------- Method 1 by @tool
# from langchain_core.tools import tool

# # Step 1: Create a function
# # Step 2: add type hinting
# # Step 3: add @tool at top of function so that llm can interact with that function

# @tool
# def multiply(a: int, b: int) -> int:
#     """Multipying two numbers"""
#     return a*b

# result = multiply.invoke({"a": 3, "b": 5})
# print(result)

# print(multiply.name)
# print(multiply.description)
# print(multiply.args)



# # -------------------------------- Method 2: Using structured tool
# from langchain_core.tools import StructuredTool
# from pydantic import BaseModel, Field

# class MultiplyInput(BaseModel):
#     a: int = Field(required=True, description="First number to multiply")
#     b: int = Field(required=True, description="Second number to multiply")
    
# def multiply(a: int, b: int) -> int:
#     return a*b

# multiply_tool = StructuredTool(
#     func=multiply,
#     name="multiply",
#     description="Multiplying two numbers",
#     args_schema=MultiplyInput
# )

# result = multiply_tool.invoke({"a": 3, "b": 5})
# print(result)



# ------------------------------- Method 3: BaseTool class
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="First number to multiply")
    b: int = Field(required=True, description="Second number to multiply")

class MultiplyTool(BaseTool):
    name: str = "Multiply"
    description: str = "Multiplication of two number"
    args_schema: Type[BaseModel] = MultiplyInput
    def _run(self, a:int, b:int) -> int:
        return a*b

multiply_tool = MultiplyTool()
result = multiply_tool.invoke({"a": 3, "b": 5})
print(result)



