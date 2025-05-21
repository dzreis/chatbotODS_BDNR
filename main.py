# Inicia a aplicação
import logging
import streamlit as st
from chat.retriever_chain import build_retriever_chain
from config import modelo_llm

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

st.title("📄 Chatbot de ajuda")

# Estado da sessão
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Inicializa cadeia de QA (Lazy)
@st.cache_resource
def init_chain():
    return build_retriever_chain(modelo_llm)

qa_chain = init_chain()

# Entrada do usuário
user_input = st.chat_input("Digite sua pergunta:")

if user_input:
    # Adiciona a pergunta ao histórico
    st.session_state.chat_history.append({"role": "user", "text": user_input})

    # Gera resposta
    response = qa_chain.invoke(user_input)
    answer = response["result"]

    # Adiciona a resposta ao histórico
    st.session_state.chat_history.append({"role": "bot", "text": answer})

# Exibe histórico
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["text"])
    else:
        st.chat_message("assistant").write(msg["text"])
