from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

load_dotenv()

docs = [
    Document(
        page_content="""
        Virat Kohli is one of the most celebrated cricketers in the world.
        He played as the captain for Royal Challengers Bangalore in the IPL.
        Known for his aggressive batting style, Kohli has broken numerous records in international cricket.
        He also led India to a historic series win in Australia in 2018–19.
        """,
        metadata={"source": "sports_1"}
    ),
    Document(
        page_content="""
        The Eiffel Tower is located in Paris, France, and is one of the most visited monuments in the world.
        Built in 1889 for the World's Fair, it stands 324 meters tall.
        Tourists often enjoy panoramic views of the city from its observation decks.
        """,
        metadata={"source": "travel_1"}
    ),
    Document(
        page_content="""
        Photosynthesis is the process by which plants produce energy from sunlight.
        It involves the conversion of carbon dioxide and water into glucose and oxygen.
        This process takes place mainly in the leaves of green plants.
        """,
        metadata={"source": "science_1"}
    ),
    Document(
        page_content="""
        Albert Einstein was a theoretical physicist best known for his theory of relativity.
        His famous equation E=mc² explains the relationship between mass and energy.
        Einstein was awarded the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect.
        """,
        metadata={"source": "history_1"}
    ),
    Document(
        page_content="""
        Drinking enough water daily is essential for maintaining hydration and overall health.
        Proper hydration helps regulate body temperature, improve mood, and support digestion.
        Experts recommend around 2–3 liters of water per day for adults, depending on activity and climate.
        """,
        metadata={"source": "health_1"}
    )
]

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory="chromaDB",
    collection_name="sample3"
)

vector_store.add_documents(docs)

base_reteriver = vector_store.as_retriever(search_kwargs={"k": 5})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
compressor = LLMChainExtractor.from_llm(llm)

compressor_reteriver = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_reteriver
)

result = compressor_reteriver.invoke("tell em about Kohli's cricket career")

print(result)







