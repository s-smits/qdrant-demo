"""
Cross-encoder reranking for improved precision.

Rerankers process query-document pairs jointly for more accurate
relevance scoring than bi-encoder embeddings.
"""

from typing import Optional
from dataclasses import dataclass

import torch
from sentence_transformers import CrossEncoder


@dataclass
class RerankResult:
    """Result after reranking."""
    id: str
    text: str
    rerank_score: float
    original_score: float
    original_rank: int
    metadata: dict


class CrossEncoderReranker:
    """
    Rerank documents using a cross-encoder model.
    
    Cross-encoders process query and document together, enabling
    deeper interaction than separate embeddings.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "auto",
        max_length: int = 512,
    ):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model name
            device: Device for inference
            max_length: Maximum sequence length
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.model_name = model_name
        self.max_length = max_length
        
        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=device,
        )
    
    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: Optional[int] = None,
    ) -> list[RerankResult]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: Search query
            documents: List of dicts with 'text', 'id', 'score', 'metadata'
            top_k: Number of results to return (None = all)
            
        Returns:
            List of RerankResult sorted by rerank_score
        """
        if not documents:
            return []
        
        # Prepare pairs
        pairs = [(query, doc["text"]) for doc in documents]
        
        # Score all pairs
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Build results
        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            results.append(RerankResult(
                id=doc.get("id", str(i)),
                text=doc["text"],
                rerank_score=float(score),
                original_score=doc.get("score", 0.0),
                original_rank=i + 1,
                metadata=doc.get("metadata", {}),
            ))
        
        # Sort by rerank score
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        if top_k:
            results = results[:top_k]
        
        return results
    
    def rerank_hybrid_results(
        self,
        query: str,
        hybrid_results: list,  # list[HybridResult]
        top_k: Optional[int] = None,
    ) -> list[RerankResult]:
        """
        Rerank HybridResult objects from hybrid retrieval.
        
        Args:
            query: Search query
            hybrid_results: Results from HybridRetriever
            top_k: Number of results to return
            
        Returns:
            Reranked results
        """
        documents = [
            {
                "id": r.id,
                "text": r.text,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in hybrid_results
        ]
        
        return self.rerank(query, documents, top_k)


class LightweightReranker:
    """
    Lightweight reranker using embedding similarity rescoring.
    
    Faster than cross-encoders but less accurate.
    Useful when cross-encoder is too slow.
    """
    
    def __init__(self, embedding_model):
        """
        Initialize with embedding model.
        
        Args:
            embedding_model: ModernEmbeddings instance
        """
        self.embeddings = embedding_model
    
    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: Optional[int] = None,
    ) -> list[RerankResult]:
        """
        Rerank using embedding similarity.
        
        Args:
            query: Search query
            documents: List of document dicts
            top_k: Number of results
            
        Returns:
            Reranked results
        """
        if not documents:
            return []
        
        # Embed query and documents
        query_emb = self.embeddings.encode_query(query).dense[0]
        doc_texts = [doc["text"] for doc in documents]
        doc_embs = self.embeddings.encode(doc_texts).dense
        
        # Compute similarities
        similarities = self.embeddings.similarity(query_emb, doc_embs)
        
        # Build results
        results = []
        for i, (doc, sim) in enumerate(zip(documents, similarities)):
            results.append(RerankResult(
                id=doc.get("id", str(i)),
                text=doc["text"],
                rerank_score=float(sim),
                original_score=doc.get("score", 0.0),
                original_rank=i + 1,
                metadata=doc.get("metadata", {}),
            ))
        
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        if top_k:
            results = results[:top_k]
        
        return results
