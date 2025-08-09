# used to load or extract text content from webpages
# is uses the beautifulSoap under the hood to parse the HTML


from langchain_community.document_loaders import WebBaseLoader

url="https://medium.com/data-science-at-microsoft/how-large-language-models-work-91c362f5b78f"
loader = WebBaseLoader(url)

docs = loader.load()
print(docs)
print(len(docs))
print(docs[0].page_content)