# using the Recursive Character text splitter but on like markdownn files, .py file, codes. etc.

from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text= """from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1️⃣ Load the document
loader = TextLoader("sample.txt", encoding="utf-8")
documents = loader.load()

# 2️⃣ Create a document-based text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,       # Max size of each chunk
    chunk_overlap=50      # Overlap between chunks
)

# 3️⃣ Split the document
splits = splitter.split_documents(documents)

# 4️⃣ Print chunks
for i, chunk in enumerate(splits, start=1):
    print(f"--- Chunk {i} ---")
    print(chunk.page_content)
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

print(chunks)
