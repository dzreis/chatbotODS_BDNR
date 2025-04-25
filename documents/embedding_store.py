# Salva os embeddings no banco
from langchain.vectorstores import Chroma
from langchain.schema.document import Document
import logging

def embeddar(chunks, persist_directory, embedding_model):
    """
    Gera embeddings para os chunks e os armazena no ChromaDB local.
    """
    documents = [
        Document(page_content=chunk["content"], metadata=chunk["metadata"])
        for chunk in chunks
    ]

    vectordb = Chroma.from_documents(
        documents,
        embedding_model,
        persist_directory=persist_directory
    )
    vectordb.persist()
    logging.info(f"Embeddings armazenados em: {persist_directory}")
