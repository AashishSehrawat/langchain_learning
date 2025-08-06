from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=300)

documents = [
    "Virat Kohli is a famous Indian cricketer known for his aggressive batting style.",
    "Sachin Tendulkar is considered one of the greatest batsmen in cricket history.",
    "Rohit Sharma holds the record for the highest individual score in One Day Internationals.",
    "MS Dhoni is renowned for his captaincy and finishing skills in limited-overs cricket.",
    "Kane Williamson is the captain of the New Zealand cricket team and known for his calm demeanor."
]

query = "Tell me about virat kohli and his batting style."

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(query)
print(f"Most similar document: {documents[index]}")
print(f"Similarity score: {score}")
