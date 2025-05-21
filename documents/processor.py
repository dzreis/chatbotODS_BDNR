# Funções para extrair e chunkar os textos
import logging
import os
import json
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("initialize_documents.log"),
        logging.StreamHandler()
    ]
)

def extract_text(folder_path):
    """
    Extrai texto de todos os arquivos PDF da pasta.
    Retorna uma lista de dicionários com 'source' e 'content'.
    """
    docs = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            try:
                with fitz.open(file_path) as pdf:
                    text = ""
                    for page in pdf:
                        text += page.get_text()
                    docs.append({"source": filename, "content": text})
            except Exception as e:
                logging.warning(f"Erro ao processar {filename}: {e}")
    return docs

def chunking(docs, chunk_size=500, chunk_overlap=50):
    """
    Divide o conteúdo dos documentos em pedaços (chunks) usando Langchain.
    Divide a cada 500 caractere com overlap de 50.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunked = []
    for doc in docs:
        splits = splitter.split_text(doc["content"])
        for i, chunk in enumerate(splits):
            chunked.append({
                "content": chunk,
                "metadata": {"source": doc["source"], "chunk_id": i}
            })
    return chunked

def save_jsonl(jsonl_path):
    """
    Carrega os chunks salvos em um arquivo JSONL.
    """
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line.strip()))
    return chunks
