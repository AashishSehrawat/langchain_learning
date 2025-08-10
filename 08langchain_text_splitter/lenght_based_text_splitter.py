# ---------------------------- text splitting of text

# from langchain.text_splitter import CharacterTextSplitter

# text = """
# In the context of LangChain and other large language model (LLM) pipelines, a text splitter is a utility designed to break large pieces of text into smaller, more manageable chunks. This process is essential because most language models have a token limit — they can only process a fixed number of tokens in a single request. If you try to feed an extremely large document into the model without splitting it, the model will either reject it, truncate it, or fail to process it effectively.

# Text splitters ensure that the content fed into the model stays within its token limit while maintaining logical coherence. The goal is not only to divide text arbitrarily but to do so in a way that preserves meaning and context as much as possible. For example, if you are splitting an academic article, you might prefer to split it at paragraph or sentence boundaries rather than in the middle of a sentence.

# The most basic type of text splitter is a character-based splitter. This method works by simply counting characters until a predefined chunk size is reached, then creating a chunk. However, this method can break sentences awkwardly. To improve this, more advanced splitters, like the RecursiveCharacterTextSplitter in LangChain, try to split at natural boundaries such as newlines, paragraphs, or sentence breaks before falling back to character splitting if necessary.

# Another common type is the token-based splitter. Instead of counting characters, it counts tokens — the units of text used by the model’s tokenizer. Since different words can have different token counts, token-based splitting is often more accurate when working directly with LLMs, ensuring that each chunk is within the model’s token limit.

# An important concept in text splitting is chunk overlap. Overlap refers to including some repeated text from the end of one chunk at the beginning of the next chunk. This is done so that context is preserved between chunks. Without overlap, information at the boundary between two chunks might be lost, leading to degraded performance in retrieval-augmented generation (RAG) or other multi-chunk workflows.

# Text splitting plays a crucial role in retrieval systems. When combined with a vector store, each chunk is converted into an embedding and stored. Later, when a user asks a question, the system retrieves the most relevant chunks based on semantic similarity and passes them to the LLM to generate an answer. Without proper splitting, important context might be scattered across chunks in a way that reduces retrieval quality.

# In LangChain, you can choose from several built-in splitters, such as CharacterTextSplitter, RecursiveCharacterTextSplitter, and TokenTextSplitter. You can also write your own custom splitter to handle special formats, like splitting a legal contract by clauses or a programming book by code examples.

# In summary, a text splitter is more than just a string cutter — it is a critical preprocessing step that balances chunk size, overlap, and semantic integrity to ensure that large documents can be efficiently and effectively processed by language models. Whether you are building a RAG system, summarizing a book, or analyzing research papers, understanding and correctly configuring a text splitter can significantly improve the quality and reliability of your AI pipeline.
# """

# splitter = CharacterTextSplitter(
#     chunk_size=100,
#     chunk_overlap=0,
#     separator=''
# )

# result = splitter.split_text(text);
# print(result)


# ------------------------------------- text splitting of pdf
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

loader = PyPDFLoader("d:/langchain/08langchain_text_splitter/genai.pdf")

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''
)

result = splitter.split_documents(docs);
print(result)
print(result[0])
print(result[0].page_content)