# Funções para extrair, limpar e chunkar os textos
import logging
import os
import json
import fitz  # PyMuPDF
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unidecode import unidecode
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Baixar os recursos necessários do NLTK
nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text, idioma='portuguese'):
    """
    Remove acentuação, stopwords e pontuação do texto.
    """
    # Remover acentuação
    text = unidecode(text)
    
    # Tokenização e filtragem
    stop_words = set(stopwords.words(idioma))
    tokens = word_tokenize(text.lower(), language=idioma)
    tokens_filtrados = [
        palavra for palavra in tokens
        if palavra not in stop_words and palavra not in string.punctuation
    ]
    return " ".join(tokens_filtrados)

def extract_text(folder_path):
    """
    Extrai texto de todos os arquivos PDF da pasta.
    Retorna uma lista de dicionários com 'source' e 'content' (pré-processado).
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
                    texto_preprocessado = preprocess_text(text)
                    docs.append({"source": filename, "content": texto_preprocessado})
            except Exception as e:
                logging.warning(f"Erro ao processar {filename}: {e}")
    return docs

def chunking(docs, chunk_size=500, chunk_overlap=50):
    """
    Divide o conteúdo dos documentos em pedaços (chunks) usando Langchain.
    Divide a cada 500 caracteres com overlap de 50.
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
