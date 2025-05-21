import logging
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import BaseLLM
from langchain.memory import ConversationBufferMemory  # NOVO: Memória para contexto da conversa
from langchain.schema import Document
from chat.ollama_llm import get_ollama_llm
from config import modelo, modelo_llm

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("initialize_documents.log"),
        logging.StreamHandler()
    ]
)

def load_vectorstore(persist_directory: str = "vectorstore") -> Chroma:
    """
    Carrega o banco vetorial persistido usando ChromaDB com embeddings do HuggingFace.
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

def build_retriever_chain(modelo_llm: str, persist_directory: str = "vectorstore") -> RetrievalQA:
    """
    Cria a cadeia de QA usando o modelo Ollama LLM + Chroma como retriever, com memória de conversa.
    """
    try:
        llm: BaseLLM = get_ollama_llm(modelo_llm)
        vectordb: Chroma = load_vectorstore(persist_directory)
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})

        # Adiciona a memória de buffer (histórico de chat)
        memory = ConversationBufferMemory(
            memory_key="chat_history",  # padrão usado pelo LangChain
            return_messages=True
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type="stuff",
            memory=memory  # AQUI usamos a memória
        )
        logging.info("Cadeia de QA com retriever e memória criada com sucesso.")
        return qa_chain
    except Exception as e:
        logging.error(f"Erro ao criar a cadeia de QA: {e}")
        raise
