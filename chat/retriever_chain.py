import logging
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import BaseLLM
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from chat.ollama_llm import get_ollama_llm
from config import modelo, modelo_llm

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

def build_retriever_chain(modelo_llm: str, memory: ConversationBufferMemory,persist_directory: str = "vectorstore") -> ConversationalRetrievalChain:
    """
    Cria a cadeia de QA usando o modelo Ollama LLM + Chroma como retriever, com memória de conversa.
    """
    try:
        print(f"Carregando modelo LLM: {modelo_llm}")
        llm: BaseLLM = get_ollama_llm(modelo_llm)
        vectordb: Chroma = load_vectorstore(persist_directory)
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})

        # Prompt estruturado
        prompt_template = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template="""
Você é uma assistente especialista em análise de documentos.

Sua tarefa é analisar o contexto fornecido e responder à pergunta do usuário com clareza, **apenas com base nas informações disponíveis no contexto**.

Use internamente técnicas como **Cadeia de Raciocínio (Chain-of-Thought)** e **Autorreflexão (Self-Reflection)** para garantir que a resposta final seja correta, mas **não exiba esses passos para o usuário**.

---

### Regras:
1. **Responda no mesmo idioma da pergunta do usuário**.
2. Não invente informações — use apenas o que estiver no contexto.
3. Se não houver dados suficientes, diga:  
   **"Não encontrei informações suficientes no contexto fornecido."**
4. A resposta deve ser clara, objetiva e bem estruturada. Use parágrafos curtos ou bullet points.

---
Histórico da Conversa:
{chat_history}

Contexto Atual:
{context}

Pergunta Atual:
{question}
"""
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt_template},
            return_source_documents=True,
            verbose=True
        )

        logging.info("Cadeia de QA com retriever, memória e prompt customizado criada com sucesso.")
        return qa_chain

    except Exception as e:
        logging.error(f"Erro ao criar a cadeia de QA: {e}")
        raise
