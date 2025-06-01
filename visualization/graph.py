import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime, timedelta
from collections import Counter
import streamlit as st
import nltk
import spacy
import difflib
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

nltk.download("stopwords")
from nltk.corpus import stopwords

# Carregando modelo de linguagem do spaCy
nlp = spacy.load("pt_core_news_sm")

# Stopwords personalizadas (pode ajustar conforme necessário)
CUSTOM_STOPWORDS = set(stopwords.words("portuguese")).union({
    "me", "minha", "qual", "mais", "foi", "última", "sobre", "pergunta", "contra", "minhas", "vez", "vezes"
})

class ChatbotMindMapGenerator:
    def __init__(self, mongo_uri: str, database_name: str, collection_name: str):
        """Initialize the ChatbotMindMapGenerator with MongoDB connection parameters."""
        self.client = MongoClient(mongo_uri)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        self.stop_words = set(stopwords.words("portuguese"))

    def fetch_chatbot_messages(self, usuario_id, limit=1000, days_back=30):
        try:
            date_filter = datetime.now() - timedelta(days=days_back)
            conversa = self.collection.find_one({
                "cod": usuario_id,
                "mensagens": {"$elemMatch": {"timestamp": {"$gte": date_filter}}}
            })
            mensagens_usuario = []
            if conversa and "mensagens" in conversa:
                for msg in conversa["mensagens"]:
                    if (msg.get("tipo") == "usuario" and "texto" in msg and 
                        msg.get("timestamp", datetime.now()) >= date_filter):
                        mensagens_usuario.append({"text": msg["texto"]})
                    if len(mensagens_usuario) >= limit:
                        break
            return mensagens_usuario
        except Exception as e:
            print(f"Erro ao buscar mensagens: {e}")
            return []

    def clean_text(self, text):
        text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zà-úA-ZÀ-Ú\s]', '', text)
        text = ' '.join(text.split())
        return ' '.join([w for w in text.lower().split() if len(w) > 2])

    def preprocess_and_extract_keywords(self, mensagens, top_n=30):
        documentos = [msg["text"] for msg in mensagens if "text" in msg]
        palavras_filtradas = []
        for doc in documentos:
            doc_spacy = nlp(doc.lower())
            palavras_doc = [
                token.lemma_ for token in doc_spacy
                if token.is_alpha and token.lemma_ not in CUSTOM_STOPWORDS and token.pos_ in {"NOUN", "VERB"}
            ]
            palavras_filtradas.append(" ".join(palavras_doc))

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(palavras_filtradas)
        palavras = vectorizer.get_feature_names_out()
        scores = X.toarray().sum(axis=0)

        palavras_score = {palavra: pontuacao for palavra, pontuacao in zip(palavras, scores)}
        palavras_importantes = sorted(palavras_score.items(), key=lambda x: x[1], reverse=True)[:top_n]
        palavras_top = [palavra for palavra, _ in palavras_importantes]

        grupos = {}
        for palavra in palavras_top:
            encontrada = False
            for chave in grupos:
                if difflib.SequenceMatcher(None, palavra, chave).ratio() > 0.85:
                    grupos[chave].append(palavra)
                    encontrada = True
                    break
            if not encontrada:
                grupos[palavra] = [palavra]

        palavras_finais = [min(grupo, key=len) for grupo in grupos.values()]
        return palavras_finais, palavras_score

    def calculate_word_similarity(self, keywords):
        if len(keywords) < 2:
            return np.array([[1]])
        contexts = []
        for keyword in keywords:
            context_texts = []
            regex_pattern = re.compile(keyword, re.IGNORECASE)
            matching_messages = self.collection.find({"mensagens.texto": {"$regex": regex_pattern}}).limit(20)
            for msg_doc in matching_messages:
                for msg in msg_doc.get("mensagens", []):
                    if keyword.lower() in msg.get("texto", "").lower():
                        context_texts.append(self.clean_text(msg["texto"]))
            contexts.append(' '.join(context_texts) if context_texts else keyword)

        vectorizer = TfidfVectorizer(stop_words=list(self.stop_words))
        try:
            tfidf_matrix = vectorizer.fit_transform(contexts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            similarity_matrix[similarity_matrix < 0.1] = 0
            return similarity_matrix
        except Exception as e:
            print(f"Erro no cálculo de similaridade: {e}")
            return np.eye(len(keywords))

    def build_graph(self, keywords, similarity_matrix, threshold=0.1):
        G = nx.Graph()
        for i, keyword in enumerate(keywords):
            G.add_node(i, label=keyword)
        n = len(keywords)
        for i in range(n):
            for j in range(i + 1, n):
                weight = similarity_matrix[i][j]
                if weight >= threshold:
                    G.add_edge(i, j, weight=weight)

        G.remove_nodes_from([node for node, degree in dict(G.degree()).items() if degree == 0])
        return G

    def categorize_keywords(self, keywords, palavras_score):
        """Categoriza as palavras-chave por importância e tipo"""
        # Obter scores das palavras
        scores = [palavras_score.get(kw, 0) for kw in keywords]
        
        # Definir categorias baseadas em quartis
        q75, q50, q25 = np.percentile(scores, [75, 50, 25])
        
        categories = {}
        for i, (kw, score) in enumerate(zip(keywords, scores)):
            if score >= q75:
                categories[i] = 'high'
            elif score >= q50:
                categories[i] = 'medium'
            elif score >= q25:
                categories[i] = 'low'
            else:
                categories[i] = 'minimal'
        
        return categories

    def get_node_colors_and_sizes(self, G, categories):
        """Define cores e tamanhos dos nós baseados na categoria"""
        color_map = {
            'high': '#FF6B6B',      # Vermelho vibrante
            'medium': '#4ECDC4',    # Turquesa
            'low': '#45B7D1',       # Azul
            'minimal': '#96CEB4'    # Verde claro
        }
        
        size_map = {
            'high': 2000,
            'medium': 1500,
            'low': 1000,
            'minimal': 700
        }
        
        node_colors = [color_map.get(categories.get(node, 'minimal'), '#96CEB4') for node in G.nodes()]
        node_sizes = [size_map.get(categories.get(node, 'minimal'), 700) for node in G.nodes()]
        
        return node_colors, node_sizes

    def visualize_graph_streamlit(self, G, keywords, palavras_score=None):
        """Visualização simplificada do mapa mental - estilo da imagem"""
        st.title("Mapa Mental do Chatbot")
        st.caption("Principais temas das conversas")
        
        # Se não há nós suficientes, usar layout simples
        if len(G.nodes()) == 0:
            st.warning("Nenhum dado disponível para visualização")
            return
        
        # Criar figura com fundo claro
        fig, ax = plt.subplots(figsize=(14, 10))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        
        # Cores simples e atrativas (palette similar à imagem)
        cores_disponiveis = [
            '#FF6B9D',  # Rosa
            '#4ECDC4',  # Turquesa  
            '#45B7D1',  # Azul
            '#96CEB4',  # Verde
            '#FECA57',  # Amarelo
            '#A29BFE',  # Roxo claro
            '#FD79A8',  # Rosa claro
            '#00CEC9',  # Ciano
            '#6C5CE7',  # Púrpura
            '#FDCB6E'   # Laranja claro
        ]
        
        # Layout circular ao redor do centro
        if len(G.nodes()) == 1:
            pos = {list(G.nodes())[0]: (2, 0)}
        else:
            pos = nx.circular_layout(G, scale=4)
        
        # Desenhar linhas pontilhadas conectando ao centro
        center = (0, 0)
        for node in pos:
            x_vals = [center[0], pos[node][0]]
            y_vals = [center[1], pos[node][1]]
            ax.plot(x_vals, y_vals, ':', color="#FDFEFF", alpha=0.7, linewidth=2)
        
        # Desenhar círculo central
        central_circle = plt.Circle((0, 0), 1.2, 
                                color='white', 
                                alpha=1,
                                zorder=2,
                                linewidth=3,
                                edgecolor="#E3F1FF")
        ax.add_patch(central_circle)
        
        ax.text(0, 0, 'PRINCIPAIS\nTEMAS DAS\nCONVERSAS', 
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            color='#2C3E50',
            zorder=4)
        
        # Desenhar os nós como círculos coloridos
        for i, (node, (x, y)) in enumerate(pos.items()):
            cor = cores_disponiveis[i % len(cores_disponiveis)]
            
            # Determinar tamanho baseado na importância (se disponível)
            if palavras_score:
                palavra = keywords[node]
                score = palavras_score.get(palavra, 0)
                scores = list(palavras_score.values())
                if scores:
                    normalized_score = (score - min(scores)) / (max(scores) - min(scores)) if max(scores) != min(scores) else 0.5
                    raio = 0.5 + (normalized_score * 0.8)  # Raio entre 0.5 e 1.3
                else:
                    raio = 0.8
            else:
                raio = 0.8
            
            # Círculo colorido
            circle = plt.Circle((x, y), raio, 
                            color=cor, 
                            alpha=0.85,
                            zorder=3)
            ax.add_patch(circle)
            
            # Texto da palavra-chave
            palavra = keywords[node].upper()
            
            # Quebrar palavras longas em múltiplas linhas
            if len(palavra) > 10:
                palavras = palavra.split()
                if len(palavras) > 1:
                    meio = len(palavras) // 2
                    palavra = '\n'.join([' '.join(palavras[:meio]), ' '.join(palavras[meio:])])
            
            ax.text(x, y, palavra, 
                ha='center', va='center',
                fontsize=8, fontweight='bold',
                color='white',
                zorder=4)
        
        # Configurar limites e aspecto
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    def run_full_analysis(self, usuario_id, limit=1000, days_back=30):
        print("🚀 Iniciando análise do chatbot...")
        messages = self.fetch_chatbot_messages(usuario_id, limit, days_back)
        if not messages:
            print("❌ Nenhuma mensagem encontrada!")
            return None, None, None
        
        keywords, palavras_score = self.preprocess_and_extract_keywords(messages)
        if not keywords:
            print("❌ Nenhuma palavra-chave extraída!")
            return None, None, None
            
        print("🔍 Calculando similaridades...")
        similarity_matrix = self.calculate_word_similarity(keywords)
        self._last_similarity_matrix = similarity_matrix  # Salvar para uso posterior
        
        print("🕸️ Criando grafo...")
        G = self.build_graph(keywords, similarity_matrix)
        
        print("✅ Análise concluída!")
        return G, keywords, similarity_matrix, palavras_score