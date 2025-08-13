from langchain_community.retrievers import WikipediaRetriever

reterivers = WikipediaRetriever(top_k_results=1, lang="en")

querry = "the history of india and pakistan from the perspective of a chinese"

docs = reterivers.invoke(querry)
print(docs)