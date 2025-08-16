# Build In tool duckduckgo-search

from langchain_community.tools import DuckDuckGoSearchRun
search_tools = DuckDuckGoSearchRun()
result_duckduck = search_tools.invoke("top news in india")
# print(result_duckduck)

# Build in tool ShellTool

from langchain_community.tools.shell.tool import ShellTool
shell = ShellTool()
result = shell.invoke("whoami")
print(result)


