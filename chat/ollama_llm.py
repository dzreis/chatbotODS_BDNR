# Conecta com os modelos de LLM do Ollama
from langchain_community.llms import Ollama
import logging
import subprocess
import json

model_name = 'llama3.2'

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

def get_ollama_llm(model_name="llama3", temperature=0.1):
    """
    Retorna um objeto LLM integrado ao Langchain usando um modelo do Ollama.
    """
    available_models = list_ollama_models()
    if model_name not in available_models:
        logging.warning(f"Modelo '{model_name}' não encontrado no Ollama. Modelos disponíveis: {available_models}")
        raise ValueError(f"Modelo '{model_name}' não está disponível no Ollama local.")

    try:
        llm = Ollama(model=model_name, temperature=temperature)
        logging.info(f"Modelo LLM '{model_name}' carregado com sucesso via Ollama.")
        return llm
    except Exception as e:
        logging.error(f"Erro ao inicializar o modelo '{model_name}' com Ollama: {e}")
        raise
