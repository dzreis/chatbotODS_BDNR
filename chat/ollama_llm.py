import logging
import subprocess
from langchain_community.llms.ollama import Ollama  # Certifique-se de usar o correto
from config import modelo_llm  # Certifique-se de que modelo_llm é uma string com o nome do modelo

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("initialize_documents.log"),
        logging.StreamHandler()
    ]
)

def list_ollama_models():
    """
    Lista os modelos disponíveis localmente no Ollama.
    Executa o comando 'ollama list' e extrai os nomes dos modelos.
    """
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().splitlines()

        # Ignora o cabeçalho e pega o nome da primeira coluna
        models = []
        for line in lines[1:]:
            parts = line.strip().split()
            if parts:
                models.append(parts[0])
        logging.info(f"Modelos disponíveis no Ollama: {models}")
        return models

    except subprocess.CalledProcessError as e:
        logging.error("Erro ao executar 'ollama list'. Verifique se o Ollama está instalado e em execução.")
        return []
    except Exception as e:
        logging.error(f"Erro inesperado ao listar modelos do Ollama: {e}")
        return []

def get_ollama_llm(modelo_llm: str, temperature: float = 0.1) -> Ollama:
    """
    Retorna uma instância da LLM conectada ao modelo local do Ollama.
    """
    available_models = list_ollama_models()
    if modelo_llm not in available_models:
        logging.warning(f"Modelo '{modelo_llm}' não encontrado no Ollama. Modelos disponíveis: {available_models}")
        raise ValueError(f"Modelo '{modelo_llm}' não está disponível localmente no Ollama.")

    try:
        llm = Ollama(model=modelo_llm, temperature=temperature)
        logging.info(f"Modelo '{modelo_llm}' carregado com sucesso via Ollama.")
        return llm
    except Exception as e:
        logging.error(f"Erro ao inicializar o modelo '{modelo_llm}': {e}")
        raise
