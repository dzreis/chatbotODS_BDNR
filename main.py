import logging
import os
import streamlit as st
from chat.retriever_chain import build_retriever_chain
from config import modelo_llm
from db.mongo_client import armazenar_conversas
from db.login import show_login_page
from dotenv import load_dotenv
from visualization.graph import ChatbotMindMapGenerator
import matplotlib.pyplot as plt
import networkx as nx

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("initialize_documents.log"),
        logging.StreamHandler()
    ]
)

# Configuração da página
st.set_page_config(page_title="Combate a violência contra mulheres", layout="wide")

# Inicialização do estado da sessão
if "user" not in st.session_state:
    st.session_state.user = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Mostra página de login se usuário não estiver logado
if not st.session_state.user:
    show_login_page()
else:
    st.title("📄🔒 ProtegeEla")
    st.subheader("Se você está sofrendo violência ou conhece alguém nessa situação, ligue para o 180 e busque ajuda — sua vida e segurança importam.")
    
    # Mostrar informações do usuário
    st.sidebar.write(f"Usuário: {st.session_state.user['nome']}")
    if st.sidebar.button("Sair"):
        st.session_state.user = None
        st.session_state.chat_history = []
        st.rerun()

    # Inicializa cadeia de QA
    @st.cache_resource
    def init_chain():
        return build_retriever_chain(modelo_llm)

    qa_chain = init_chain()

    # Interface do chat
    user_input = st.chat_input("Digite sua pergunta:")

    if user_input:
        # Adiciona pergunta ao histórico
        st.session_state.chat_history.append({"role": "user", "text": user_input})

        try:
            # Testando diferentes formatos de entrada
            try:
                # Primeiro tenta com 'question' (padrão do RetrievalQA)
                response = qa_chain({"question": user_input})
            except Exception as e1:
                logging.info(f"Tentativa com 'question' falhou: {e1}")
                try:
                    # Se falhar, tenta com 'query'
                    response = qa_chain({"query": user_input})
                except Exception as e2:
                    logging.info(f"Tentativa com 'query' falhou: {e2}")
                    # Se ambos falharem, tenta passagem direta
                    response = qa_chain(user_input)
            
            # Processa a resposta
            if response:
                if isinstance(response, dict) and "result" in response:
                    answer = response["result"]
                elif isinstance(response, dict) and "answer" in response:
                    answer = response["answer"]
                elif isinstance(response, str):
                    answer = response
                else:
                    answer = str(response)
            else:
                answer = "Desculpe, não consegui processar sua pergunta."

            # Adiciona resposta ao histórico
            st.session_state.chat_history.append({"role": "bot", "text": answer})

            # Salva conversa no banco
            armazenar_conversas(
                None, 
                st.session_state.user["id"],
                user_input,
                answer
            )
            
        except Exception as e:
            error_msg = f"Erro ao gerar resposta: {str(e)}"
            st.error(error_msg)
            logging.error(error_msg)
            
            # Adiciona mensagem de erro ao histórico para debug
            st.session_state.chat_history.append({
                "role": "bot", 
                "text": "Desculpe, ocorreu um erro interno. Tente reformular sua pergunta."
            })

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["text"])
        else:
            st.chat_message("assistant").write(msg["text"])

    # Adiciona separador visual
    st.markdown("---")

    # Seção de análise do grafo
    st.sidebar.markdown("## Análise de Conversas")
    if st.sidebar.button("Gerar Análise"):
        load_dotenv()
        mongo_uri = os.getenv("MONGO_URI")
        mindmap_generator = ChatbotMindMapGenerator(mongo_uri, "chat_bot", "conversas")
        G, keywords, similarity_matrix = mindmap_generator.run_full_analysis(
            usuario_id=st.session_state.user["id"],
            limit=500,
            days_back=30
        )
        if G is not None and keywords is not None and similarity_matrix is not None:
            st.subheader("Mapa Mental das Conversas")
            mindmap_generator.visualize_graph_streamlit(G, keywords)
        else:
            st.warning("Não foi possível gerar a análise. Verifique se existem mensagens suficientes.")