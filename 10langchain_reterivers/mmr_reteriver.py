# this reteriver pick the results that are not only relevent to query but differnt from each other

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

reteriver = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3, "lambda_mult":0}
)

query="What are the teams name form which Virat Kholi, MS Dhoni, Rohit Sharma play?"
result = reteriver.invoke(query)
print(result)


