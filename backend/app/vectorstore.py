import chromadb
from .config import settings

client = chromadb.Client(
    settings=chromadb.config.Settings(
        persist_directory=settings.VECTOR_DB_PATH
    )
)

collection = client.get_or_create_collection("documents")

def add_documents(chunks, embeddings):
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i]],
            ids=[f"id_{hash(chunk)}"]
        )

def query_documents(query_embedding):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    return results