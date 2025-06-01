import logging
import os
import streamlit as st
from langchain.memory import ConversationBufferMemory
from chat.retriever_chain import build_retriever_chain
from config import modelo_llm
from db.mongo_client import armazenar_conversas
from db.login import show_login_page
from dotenv import load_dotenv
from visualization.graph import ChatbotMindMapGenerator

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("initialize_documents.log"),
        logging.StreamHandler()
    ]
)

# Configuração da página Streamlit
st.set_page_config(page_title="Sua vida e segurança importam", layout="wide")

# Estado da sessão
if "user" not in st.session_state:
    st.session_state.user = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )

@st.cache_resource(show_spinner="Carregando inteligência do chatbot...")
def init_chain():
    return build_retriever_chain(modelo_llm, memory=st.session_state.memory)

# Lógica principal
if not st.session_state.user:
    show_login_page()
else:
    st.title("📢 ProtegeEla")
    st.subheader("Se você está sofrendo violência ou conhece alguém nessa situação, busque ajuda - disque 180")

    # Sidebar - informações do usuário
    st.sidebar.write(f"Usuário: {st.session_state.user['nome']}")
    if st.sidebar.button("Sair"):
        st.session_state.user = None
        st.session_state.chat_history = []
        st.rerun()

    qa_chain = init_chain()

    # Entrada do chat
    user_input = st.chat_input("Digite sua pergunta:")

    if user_input:
        try:
            st.session_state.chat_history.append({"role": "user", "text": user_input})

            response = qa_chain({
                "question": user_input,
                "chat_history": [msg["text"] for msg in st.session_state.chat_history]
            })

            if response and "answer" in response:
                answer = response["answer"]
                st.session_state.chat_history.append({"role": "bot", "text": answer})

                armazenar_conversas(
                    None,
                    st.session_state.user["id"],
                    user_input,
                    answer
                )

        except Exception as e:
            logging.error("Erro ao gerar resposta: %s", str(e))
            st.error("Desculpe, ocorreu um erro interno. Tente reformular sua pergunta.")
            st.session_state.chat_history.append({
                "role": "bot",
                "text": "Desculpe, ocorreu um erro interno. Tente reformular sua pergunta."
            })

    # Exibição do histórico
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["text"])
        else:
            st.chat_message("assistant").write(msg["text"])

    st.markdown("---")

    # Seção de análise do grafo
    st.sidebar.markdown("## Análise de Conversas")
    if st.sidebar.button("Gerar Análise"):
        try:
            load_dotenv()
            mongo_uri = os.getenv("MONGO_URI")
            if not mongo_uri:
                st.error("Erro: MONGO_URI não encontrada")
                st.stop()
                
            mindmap_generator = ChatbotMindMapGenerator(
                mongo_uri=mongo_uri,
                database_name="chat_bot",
                collection_name="conversas"
            )
            
            # Update to unpack 4 values instead of 3
            G, keywords, similarity_matrix, palavras_score = mindmap_generator.run_full_analysis(
                usuario_id=st.session_state.user["id"],
                limit=500,
                days_back=30
            )
            
            if G is not None and keywords is not None and similarity_matrix is not None:
                st.subheader("Mapa Mental das Conversas")
                # Passa palavras_score para visualization
                mindmap_generator.visualize_graph_streamlit(G, keywords, palavras_score)
            else:
                st.warning("Não foi possível gerar a análise. Verifique se existem mensagens suficientes.")
                
        except Exception as e:
            st.error(f"Erro ao gerar análise: {str(e)}")
            logging.error(f"Erro na geração do grafo: {str(e)}")