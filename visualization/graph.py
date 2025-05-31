import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime, timedelta
import pandas as pd
import os
from dotenv import load_dotenv
import streamlit as st

class ChatbotMindMapGenerator:
    def __init__(self, mongo_uri, database_name, collection_name):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        self.stop_words = {'o', 'a', 'os', 'as', 'de', 'da', 'do', 'das', 'dos', 'e', 'ou',
                          'mas', 'por', 'para', 'com', 'em', 'no', 'na', 'nos', 'nas',
                          'que', 'como', 'quando', 'onde', 'eu', 'tu', 'ele', 'ela',
                          'nós', 'vós', 'eles', 'elas', 'um', 'uma', 'uns', 'umas'}

    def fetch_chatbot_messages(self, usuario_id, limit=1000, days_back=30):
        try:
            date_filter = datetime.now() - timedelta(days=days_back)
            
            # Busca o documento de conversas do usuário
            conversa = self.collection.find_one({
                "cod": usuario_id,
                "mensagens": {
                    "$elemMatch": {
                        "timestamp": {"$gte": date_filter}
                    }
                }
            })
            
            mensagens_usuario = []
            
            if conversa and "mensagens" in conversa:
                # Filtra mensagens do usuário dentro do período
                for msg in conversa["mensagens"]:
                    if (msg.get("tipo") == "usuario" and 
                        "texto" in msg and 
                        msg.get("timestamp", datetime.now()) >= date_filter):
                        mensagens_usuario.append({"text": msg["texto"]})
                    if len(mensagens_usuario) >= limit:
                        break
                        
            print(f"Encontradas {len(mensagens_usuario)} mensagens de usuários")
            return mensagens_usuario

        except Exception as e:
            print(f"Erro ao buscar mensagens: {e}")
            return []

    def extract_keywords(self, messages, min_freq=2, max_features=50):
        texts = [self.clean_text(msg['text']) for msg in messages if 'text' in msg]
        if not texts:
            print("Nenhum texto encontrado nas mensagens")
            return []

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=list(self.stop_words),
            min_df=min_freq,
            ngram_range=(1, 2),
            token_pattern=r'[a-zà-ú]+',
            lowercase=True
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            word_scores = list(zip(feature_names, mean_scores))
            word_scores.sort(key=lambda x: x[1], reverse=True)
            keywords = [word for word, score in word_scores if score > 0]
            print(f"Extraídas {len(keywords)} palavras-chave relevantes")
            return keywords[:max_features]

        except Exception as e:
            print(f"Erro na extração de palavras-chave: {e}")
            return []

    def clean_text(self, text):
        text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zà-úA-ZÀ-Ú\s]', '', text)
        text = ' '.join(text.split())
        return text.lower()

    def calculate_word_similarity(self, keywords):
        if len(keywords) < 2:
            return np.array([[1]])

        contexts = []
        for keyword in keywords:
            context_texts = []
            regex_pattern = re.compile(keyword, re.IGNORECASE)
            matching_messages = self.collection.find({
                "mensagens.texto": {"$regex": regex_pattern}
            }).limit(20)

            for msg_doc in matching_messages:
                for msg in msg_doc.get("mensagens", []):
                    if keyword.lower() in msg.get("texto", "").lower():
                        context_texts.append(self.clean_text(msg["texto"]))

            if not context_texts:
                context_texts = [keyword]
            contexts.append(' '.join(context_texts))

        vectorizer = TfidfVectorizer(stop_words=list(self.stop_words))
        try:
            tfidf_matrix = vectorizer.fit_transform(contexts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            similarity_matrix[similarity_matrix < 0.1] = 0
            return similarity_matrix

        except Exception as e:
            print(f"Erro no cálculo de similaridade: {e}")
            return np.eye(len(keywords))

    def create_mindmap_graph(self, keywords, similarity_matrix):
        G = nx.Graph()
        for i, keyword in enumerate(keywords):
            G.add_node(i, label=keyword)

        n = len(keywords)
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i][j] > 0:
                    G.add_edge(i, j, weight=similarity_matrix[i][j])

        return G

    def save_keywords(self, keywords):
        try:
            keyword_docs = [{'keyword': kw, 'created_at': datetime.now()} for kw in keywords]
            self.db['keywords'].insert_many(keyword_docs)
            print(f"✅ {len(keywords)} palavras-chave salvas na coleção 'keywords'")
        except Exception as e:
            print(f"Erro ao salvar palavras-chave: {e}")

    def save_keyword_relations(self, keywords, similarity_matrix, threshold=0.1):
        try:
            relations = []
            n = len(keywords)
            for i in range(n):
                for j in range(i + 1, n):
                    weight = similarity_matrix[i][j]
                    if weight >= threshold:
                        relations.append({
                            'source': keywords[i],
                            'target': keywords[j],
                            'weight': float(weight),
                            'created_at': datetime.now()
                        })
            if relations:
                self.db['relations'].insert_many(relations)
                print(f"✅ {len(relations)} relações salvas na coleção 'relations'")
            else:
                print("⚠️ Nenhuma relação relevante para salvar")
        except Exception as e:
            print(f"Erro ao salvar relações: {e}")

    def run_full_analysis(self, usuario_id, limit=1000, days_back=30):
        print("🚀 Iniciando análise do chatbot...")
        messages = self.fetch_chatbot_messages(usuario_id, limit, days_back)
        if not messages:
            print("❌ Nenhuma mensagem encontrada!")
            return None, None, None  # Return tuple of None values

        keywords = self.extract_keywords(messages)
        if not keywords:
            print("❌ Nenhuma palavra-chave extraída!")
            return None, None, None  # Return tuple of None values

        print("🔍 Calculando similaridades...")
        similarity_matrix = self.calculate_word_similarity(keywords)

        print("🕸️ Criando grafo...")
        G = self.create_mindmap_graph(keywords, similarity_matrix)

        self.save_keywords(keywords)
        self.save_keyword_relations(keywords, similarity_matrix)

        print("✅ Análise concluída!")
        return G, keywords, similarity_matrix
    
    def visualize_graph_streamlit(self, G, keywords):
        
        fig, ax = plt.subplots(figsize=(10, 10))
        pos = nx.spring_layout(G)
        
        nx.draw(G, pos, 
                with_labels=True,
                labels={i: keywords[i] for i in G.nodes()},
                node_color='lightblue',
                node_size=1000,
                font_size=8,
                ax=ax)
        
        st.pyplot(fig)
        plt.close()
