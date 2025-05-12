# Conecta com os modelos de LLM do Ollama
from langchain_community.llms import Ollama
import logging
import subprocess
import json
from config import modelo_llm

def list_ollama_models():
    """
    Lista os modelos disponíveis localmente no Ollama.
    """
    try:
        result = subprocess.run(["ollama", "list", "--json"], capture_output=True, text=True, check=True)
        model_data = json.loads(result.stdout)
        models = [model["name"] for model in model_data.get("models", [])]
        logging.info(f"Modelos disponíveis no Ollama: {models}")
        return models
    except subprocess.CalledProcessError as e:
        logging.error("Erro ao executar 'ollama list'. Verifique se o Ollama está instalado e em execução.")
        return []

def get_ollama_llm(modelo_llm, temperature=0.1):
    """
    Retorna um objeto LLM integrado ao Langchain usando um modelo do Ollama.
    """
    available_models = list_ollama_models()
    if modelo_llm not in available_models:
        logging.warning(f"Modelo '{modelo_llm}' não encontrado no Ollama. Modelos disponíveis: {available_models}")
        raise ValueError(f"Modelo '{modelo_llm}' não está disponível no Ollama local.")

    try:
        llm = Ollama(model=modelo_llm, temperature=temperature)
        logging.info(f"Modelo LLM '{modelo_llm}' carregado com sucesso via Ollama.")
        return llm
    except Exception as e:
        logging.error(f"Erro ao inicializar o modelo '{modelo_llm}' com Ollama: {e}")
        raise
