# vector store is alredy created and docs stored as embedding in it. 
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store = Chroma(
    embedding_function=embedding,
    persist_directory="chromaDB",
    collection_name="sample"
)

reteriver = vector_store.as_retriever(search_kwargs={"k":1})

query = "what is former RCB captain name?"
result = reteriver.invoke(query)

print(result)

