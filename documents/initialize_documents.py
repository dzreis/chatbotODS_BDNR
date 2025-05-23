# Executa o pré-processamento dos documentos
import logging
from langchain.embeddings import HuggingFaceEmbeddings
from processor import extract_text, chunking, save_jsonl
from embedding_store import embeddar

FILES_DIR = "files"
OUTPUT_JSONL = "content.jsonl"
CHROMA_DIR = "vectorstore"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if __name__ == "__main__":
    logging.info("Iniciando processamento dos arquivos PDF...")

    raw_docs = extract_text(FILES_DIR)
    logging.info(f"{len(raw_docs)} documentos extraídos.")

    chunks = chunking(raw_docs)
    logging.info(f"{len(chunks)} chunks gerados.")

    save_jsonl(chunks, OUTPUT_JSONL)

    embeddar(chunks, CHROMA_DIR, embedding_model)

    logging.info("Inicialização de documentos concluída com sucesso.")
