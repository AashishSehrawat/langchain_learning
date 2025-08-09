# used when there are multiple pdfs in directory
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="path_of_directory",
    glob="*.pdf", # pattern to which file to load
    loader_cls=PyPDFLoader
)

# docs = loader.load()
#---------------------or
docs = loader.lazy_load()
