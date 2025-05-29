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
information from the provided context and deliver an accurate, well-structured response strictly
based on the available data, following logical reasoning.\n\n

### Context:
{context}

### Instructions:
Follow a structured reasoning process to ensure accurate answers. Use **Chain-of-Thought (CoT)** and 
**Self-Reflection** techniques before finalizing your response. Your answer **must be in the same language as the question**.

#### Step 1: Understanding the Question
1. Identify the key components of the question.
2. Detect the language of the question and ensure the response is in the same language.
3. Determine if multiple interpretations exist and clarify them.

#### Step 2: Analyzing the Context (Chain-of-Thought)
4. Break down the reasoning step-by-step based on the provided context.
5. Extract relevant evidence from the context and **cite specific excerpts** when applicable.
6. If there are conflicting pieces of information, summarize both and provide a balanced conclusion.

#### Step 3: Generating a Structured Response
7. Ensure the response is **clear, concise, and logically structured**.
8. If needed, present the answer in bullet points or sections for better readability.

#### Step 4: Self-Reflection & Verification
9. Before finalizing, verify the response by asking:
   - Does it strictly adhere to the provided context?
   - Does it avoid assumptions or hallucinations?
   - Is the reasoning clear and logically sound?
   - Are relevant excerpts cited when necessary?
10. If any issue is found, refine the response before displaying it to the user.

11. If the answer is not found in the provided context, explicitly state: "I did not find sufficient information."

### User Question:
{question}

**Final Answer (respond in the same language as the question):**
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
