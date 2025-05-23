import logging
from pymongo import MongoClient
from datetime import datetime
from typing import List, Dict

# Conexão com o MongoDB
client = MongoClient("mongodb://localhost:27017/")  # Ajuste a URI conforme necessário
db = client["chatbot"]
collection = db["conversation_history"]

def salvar_interacao(usuario: str, pergunta: str, resposta: str) -> None:
    """
    Salva uma interação (pergunta e resposta) com timestamp no banco MongoDB.
    """
    try:
        doc = {
            "usuario": usuario,
            "pergunta": pergunta,
            "resposta": resposta,
            "timestamp": datetime.now()
        }
        collection.insert_one(doc)
        logging.info(f"Interação salva com sucesso para usuário: {usuario}")
    except Exception as e:
        logging.error(f"Erro ao salvar histórico: {e}")

def recuperar_historico(usuario: str, limite: int = 10) -> List[Dict]:
    """
    Recupera as últimas interações de um usuário.
    """
    try:
        historico = list(
            collection.find({"usuario": usuario}).sort("timestamp", -1).limit(limite)
        )
        logging.info(f"{len(historico)} interações recuperadas para usuário: {usuario}")
        return historico
    except Exception as e:
        logging.error(f"Erro ao recuperar histórico: {e}")
        return []

# Exemplo de uso (remova em produção)
if __name__ == "__main__":
    salvar_interacao("joao", "Qual a capital do Brasil?", "A capital do Brasil é Brasília.")
    historico = recuperar_historico("joao")
    for h in historico:
        print(f"[{h['timestamp']}] {h['pergunta']} → {h['resposta']}")