import logging
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
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

def build_retriever_chain(modelo_llm: str, persist_directory: str = "vectorstore") -> RetrievalQA:
    """
    Cria a cadeia de QA usando o modelo Ollama LLM + Chroma como retriever, com memória de conversa.
    """
    try:
        print(f"Carregando modelo LLM: {modelo_llm}")
        llm: BaseLLM = get_ollama_llm(modelo_llm)
        vectordb: Chroma = load_vectorstore(persist_directory)
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})

        # Memória da conversa
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            return_messages=True,
            output_key="answer"
        )

        # Prompt estruturado em inglês
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are an expert assistant specialized in document analysis. Your goal is to extract relevant
information from the provided context and deliver a precise, well-structured answer **only** based on the available data.

You will use internal reasoning techniques such as **Chain-of-Thought** and **Self-Reflection** to ensure the response is complete and accurate. However, **only the final answer must be shown to the user** — do not expose the reasoning steps.

---

### Context:
{context}

### Internal Reasoning (Do not show this to the user):
1. Understand the question:
   - Identify key elements.
   - Detect language and ensure response will be in the same language.
   - Clarify ambiguities if needed.

2. Analyze the context step by step:
   - Use Chain-of-Thought to find relevant evidence.
   - Cite supporting excerpts during reasoning (but not in the final output unless essential).
   - Balance conflicting data if present.

3. Generate a clear and accurate answer:
   - Use bullet points or sections if it improves clarity.
   - Keep the response concise, complete, and directly related to the question.

4. Self-Reflect before finalizing:
   - Does the response strictly follow the context?
   - Are there hallucinations or assumptions?
   - Is the logic sound and the language correct?
   - Refine if needed.

5. If the context lacks sufficient information, say:
   - "I did not find sufficient information."

---

### User Question:
{question}

**Final Answer (respond only with the final answer, in the same language as the question):**
"""
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt_template,
                "verbose": True
            }
        )

        logging.info("Cadeia de QA com retriever, memória e prompt customizado criada com sucesso.")
        return qa_chain

    except Exception as e:
        logging.error(f"Erro ao criar a cadeia de QA: {e}")
        raise
