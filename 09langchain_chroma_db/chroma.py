from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()

docs1 = Document(
    page_content="Virat Kohli is former captian RCB",
    metadata={'team': 'RCB'}
)

docs2 = Document(
    page_content="MS Dhoni is the captain of CSK",
    metadata={'team': 'CSK'}
)

docs3 = Document(
    page_content="Rohit Sharma plays for Mumbai Indians",
    metadata={'team': 'MI'}
)

docs4 = Document(
    page_content="KL Rahul leads Lucknow Super Giants",
    metadata={'team': 'LSG'}
)

docs5 = Document(
    page_content="Shubman Gill represents Gujarat Titans",
    metadata={'team': 'GT'}
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

docs = [docs1, docs2, docs3, docs4, docs5]
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="chromaDB",
    collection_name="sample"
)

# adding docs to vector db 
added_docs = vector_store.add_documents(docs)
print("Added data: ", added_docs)

data = vector_store.get(include=["embeddings", "documents", "metadatas"])
print("Data: ", data)

query = vector_store.similarity_search(
    query="who is virat kholi",  
    k=1
)
print("Query: ", query)
