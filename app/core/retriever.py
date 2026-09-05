# app/core/retriever.py
import numpy as np
from typing import List, Dict
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer

class HybridRetriever:
    """
    Гибридный поиск: BM25 + Вектора.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "intfloat/multilingual-e5-large",
        weight_bm25: float = 0.3,
        weight_vector: float = 0.7,
        top_k: int = 5
    ):
        self.weight_bm25 = weight_bm25
        self.weight_vector = weight_vector
        self.top_k = top_k
        
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        self.chunks = []
        self.bm25 = None
        self.faiss_index = None
        self.is_initialized = False
    
    def initialize(self, chunks: List[Dict]):
        """
        Инициализация индексов.
        """
        self.chunks = chunks
        
        if not chunks:
            raise ValueError("No chunks provided for initialization")
        
        texts = [chunk['text'] for chunk in chunks]
        
        # BM25
        tokenized_texts = [text.split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        
        # FAISS
        print("Generating embeddings for FAISS...")
        embeddings = self.embedding_model.encode(
            texts, 
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings.astype('float32'))
        
        self.is_initialized = True
        print(f"Initialized with {len(chunks)} chunks")
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Гибридный поиск.
        """
        if not self.is_initialized:
            raise ValueError("Retriever not initialized.")
        
        # BM25
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # FAISS
        query_embedding = self.embedding_model.encode(
            [query], 
            normalize_embeddings=True
        ).astype('float32')
        
        vector_scores, _ = self.faiss_index.search(query_embedding, len(self.chunks))
        vector_scores = vector_scores[0]
        
        # Нормализация
        bm25_normalized = self._normalize_scores(bm25_scores)
        vector_normalized = self._normalize_scores(vector_scores)
        
        # Комбинация
        combined_scores = []
        for i in range(len(self.chunks)):
            combined_score = (
                self.weight_bm25 * bm25_normalized[i] +
                self.weight_vector * vector_normalized[i]
            )
            combined_scores.append({
                'chunk': self.chunks[i],
                'score': combined_score,
                'bm25_score': bm25_scores[i],
                'vector_score': vector_scores[i] if i < len(vector_scores) else 0,
            })
        
        combined_scores.sort(key=lambda x: x['score'], reverse=True)
        
        return combined_scores[:self.top_k]
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return scores
        
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score - min_score == 0:
            return np.zeros_like(scores)
        
        return (scores - min_score) / (max_score - min_score)