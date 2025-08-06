from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

docs = [
    "Hello world",
    "This is a test document.",
    "Another example of a document to embed."
]

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=None,  # Set to None to use the key from the environment variable
    dimensions=32
)

result = embeddings.embed_documents(docs)
print(str(result)) 
