# Cadeia de geração de resposta com LangChain
import logging
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from chat.ollama_llm import get_ollama_llm
from config import modelo, modelo_llm

def load_vectorstore(persist_directory="vectorstore"):
    """
    Carrega a base vetorial persistida (ChromaDB).
    """
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=modelo)
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
        logging.info("Vectorstore carregado com sucesso.")
        return vectordb
    except Exception as e:
        logging.error(f"Erro ao carregar vectorstore: {e}")
        raise

def build_retriever_chain(modelo_llm, persist_directory="vectorstore"):
    """
    Cria a cadeia de QA usando Ollama LLM + Chroma Retriever.
    """
    llm = get_ollama_llm(modelo_llm)
    vectordb = load_vectorstore(persist_directory)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff"
    )
    logging.info("Cadeia de QA com retriever criada com sucesso.")
    return qa_chain
