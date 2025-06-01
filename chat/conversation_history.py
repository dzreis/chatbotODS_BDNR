import os
import sys
import logging
from pymongo import MongoClient
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from db.mongo_client import conectar

# Conexão com o MongoDB
db = conectar()

colecao_usuarios = db.usuarios
colecao_conversas = db.conversas

def salvar_interacao(usuario_id: str, pergunta: str, resposta: str) -> None:
    """
    Salva uma interação (pergunta e resposta) com timestamp no banco MongoDB
    na estrutura com campo 'cod' e lista 'mensagens'.
    """
    try:
        conversa = {
            "cod": usuario_id,
            "mensagens": [
                {"tipo": "usuario", "texto": pergunta, "timestamp": datetime.now()},
                {"tipo": "bot", "texto": resposta, "timestamp": datetime.now()}
            ]
        }
        colecao_conversas.insert_one(conversa)
        logging.info(f"Interação salva com sucesso para usuário: {usuario_id}")
    except Exception as e:
        logging.error(f"Erro ao salvar histórico: {e}")

def recuperar_historico(usuario: str, limite: int = 10) -> List[Dict]:
    """
    Recupera as últimas interações de um usuário.
    """
    try:
        historico = list(
            colecao_conversas.find({"cod": usuario}).sort("timestamp", -1).limit(limite)
        )
        logging.info(f"{len(historico)} interações recuperadas para usuário: {usuario}")
        return historico
    except Exception as e:
        logging.error(f"Erro ao recuperar histórico: {e}")
        return []
