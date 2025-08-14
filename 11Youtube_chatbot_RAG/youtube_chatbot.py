from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# STEP 1: INDEXING
# Step 1(a): Document Ingestion
# loading the data or transcript from youtube vedio 
video_id = "Gfr50f6ZBvo"
try:
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.list(video_id)
    try:
        transcript_obj = transcript_list.find_transcript(['en'])
    except:
        transcript_obj = next(iter(transcript_list))
    
    transcript_fetched = transcript_obj.fetch()
    transcript = " ".join(chunk.text for chunk in transcript_fetched)
    # print(transcript)
except Exception as e:
    print("Error in Youtube Transcript: ", e)
    transcript = ""

# Step 1(b): Text Splitter
# divide the transcript into small chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
# print(chunks)

# Step 1(c) &1(d): Embedding Generation and store in vector store
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_documents(chunks, embedding)
# print(vector_store.index_to_docstore_id)


# STEP: 2 Reteriver: input(query) => output(list of Documents)
reteriver = vector_store.as_retriever(serach_type="similarity", search_kwargs={"k": 4})
# print(reteriver.invoke("what is deepmind"))


# STEP: 3 Augmentation
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.4)

prompt = PromptTemplate(
    template="""
    You are an helpful assistant
    Answer only from the provided transcript context.
    if the context insufficient, just say you don't know,

    {content}
    Question: {question}
    """,
    input_variables=['content', 'question']
)

# question = "the topic of aliens discussed in this video? if yes then what was discussed"
# reteriver_docs = reteriver.invoke(question)

# context_text = "\n\n".join(docs.page_content for docs in reteriver_docs)

# final_prompt = prompt.invoke({"content": context_text, "question": question})

# # STEP: 4 Generation
# answer = llm.invoke(final_prompt);
# print(answer.content)



# By using chains
def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs) 


parallel_chain = RunnableParallel({
    'content': reteriver | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser
result = main_chain.invoke('the topic of aliens discussed in this video? if yes then what was discussed')
print(result)
