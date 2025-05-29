import streamlit as st
from db.mongo_client import cadastrar_usuario, login_usuario, get_historico_usuario

def clear_form():
    """Helper function to reset form"""
    st.session_state.form_submitted = True
    st.rerun()

def show_login_page():
    # Initialize session state for form control
    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = False

    st.title("Login / Cadastro")
    
    tab1, tab2 = st.tabs(["Login", "Cadastro"])
    
    with tab1:
        st.header("Login")
        email = st.text_input("Email", key="login_email")
        senha = st.text_input("Senha", type="password", key="login_senha")
        
        if st.button("Entrar"):
            if email and senha:
                usuario = login_usuario(email, senha)
                if usuario:
                    st.session_state.user = {
                        "id": str(usuario["_id"]),
                        "nome": usuario["nome"],
                        "email": usuario["email"]
                    }
                    st.success(f"Bem vindo, {usuario['nome']}!")
                    
                    # Carrega histórico de conversas
                    historico = get_historico_usuario(st.session_state.user["id"])
                    if historico:
                        st.session_state.chat_history = []
                        for conversa in historico:
                            for msg in conversa["mensagens"]:
                                st.session_state.chat_history.append({
                                    "role": msg["tipo"],
                                    "text": msg["texto"]
                                })
                    st.rerun()
                else:
                    st.error("Email ou senha inválidos")
    
    with tab2:
        st.header("Cadastro")
        
        # Use empty values if form was just submitted
        default_nome = "" if st.session_state.form_submitted else st.session_state.get('cadastro_nome', '')
        default_email = "" if st.session_state.form_submitted else st.session_state.get('cadastro_email', '')
        default_telefone = "" if st.session_state.form_submitted else st.session_state.get('cadastro_telefone', '')
        default_nascimento = "" if st.session_state.form_submitted else st.session_state.get('cadastro_nascimento', '')
        
        nome = st.text_input("Nome", value=default_nome, key="cadastro_nome")
        email = st.text_input("Email", value=default_email, key="cadastro_email")
        telefone = st.text_input("Telefone", value=default_telefone, key="cadastro_telefone")
        nascimento = st.text_input("Data de Nascimento (aaaa/mm/dd)", 
                                 value=default_nascimento, 
                                 key="cadastro_nascimento")
        senha = st.text_input("Senha", type="password", key="cadastro_senha")
        
        if st.button("Cadastrar"):
            if nome and email and senha and telefone and nascimento:
                user_id, msg = cadastrar_usuario(
                    nome=nome,
                    telefone=telefone,
                    senha=senha,
                    email=email,
                    nascimento=nascimento
                )
                if user_id:
                    st.success(msg)
                    # Reset form using rerun
                    clear_form()
                else:
                    st.error(msg)
            else:
                st.error("Todos os campos são obrigatórios!")
                
    # Reset form_submitted flag after rerun
    if st.session_state.form_submitted:
        st.session_state.form_submitted = False