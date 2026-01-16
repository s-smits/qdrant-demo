"""
Hybrid retrieval with Reciprocal Rank Fusion.

Combines dense vector search with sparse BM25 retrieval
for optimal recall and precision.
"""

from typing import Optional
from dataclasses import dataclass

import numpy as np

from ..vectorstore.qdrant_store import QdrantStore, BM25Store, SearchResult


@dataclass
class HybridResult:
    """Result from hybrid retrieval."""
    id: str
    text: str
    score: float
    metadata: dict
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None


class HybridRetriever:
    """
    Hybrid retrieval combining dense and sparse search with RRF.
    
    Reciprocal Rank Fusion (RRF) combines rankings from multiple
    retrieval methods without needing score normalization.
    """
    
    def __init__(
        self,
        dense_store: QdrantStore,
        sparse_store: BM25Store,
        embeddings,  # ModernEmbeddings instance
        rrf_k: int = 60,
        dense_weight: float = 0.5,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            dense_store: Qdrant vector store
            sparse_store: BM25 index
            embeddings: Embedding model for query encoding
            rrf_k: RRF constant (higher = more uniform blending)
            dense_weight: Weight for dense vs sparse (0-1)
        """
        self.dense_store = dense_store
        self.sparse_store = sparse_store
        self.embeddings = embeddings
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
    
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        dense_only: bool = False,
        sparse_only: bool = False,
    ) -> list[HybridResult]:
        """
        Retrieve documents using hybrid search with RRF.
        
        Args:
            query: Search query
            top_k: Number of results to return
            dense_only: Use only dense retrieval
            sparse_only: Use only sparse retrieval
            
        Returns:
            List of HybridResult objects
        """
        # Get more candidates for fusion
        fetch_k = top_k * 2
        
        # Dense retrieval
        dense_results = []
        if not sparse_only:
            query_result = self.embeddings.encode_query(query)
            query_embedding = query_result.dense[0]
            
            dense_results = self.dense_store.search(
                query_embedding=query_embedding,
                top_k=fetch_k,
            )
        
        # Sparse retrieval
        sparse_results = []
        if not dense_only:
            sparse_results = self.sparse_store.search(
                query=query,
                top_k=fetch_k,
            )
        
        # If only one method, return directly
        if dense_only:
            return [
                HybridResult(
                    id=r.id,
                    text=r.text,
                    score=r.score,
                    metadata=r.metadata,
                    dense_rank=i + 1,
                )
                for i, r in enumerate(dense_results[:top_k])
            ]
        
        if sparse_only:
            docs = self.sparse_store.documents
            return [
                HybridResult(
                    id=str(idx),
                    text=docs[idx]["text"],
                    score=score,
                    metadata=docs[idx].get("metadata", {}),
                    sparse_rank=i + 1,
                )
                for i, (idx, score) in enumerate(sparse_results[:top_k])
            ]
        
        # Reciprocal Rank Fusion
        rrf_scores = {}
        doc_data = {}
        
        # Score dense results
        for rank, result in enumerate(dense_results):
            doc_id = result.id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0)
            rrf_scores[doc_id] += self.dense_weight / (self.rrf_k + rank + 1)
            
            if doc_id not in doc_data:
                doc_data[doc_id] = {
                    "text": result.text,
                    "metadata": result.metadata,
                    "dense_rank": rank + 1,
                }
            else:
                doc_data[doc_id]["dense_rank"] = rank + 1
        
        # Score sparse results
        docs = self.sparse_store.documents
        for rank, (doc_idx, _) in enumerate(sparse_results):
            doc_id = str(doc_idx)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0)
            rrf_scores[doc_id] += (1 - self.dense_weight) / (self.rrf_k + rank + 1)
            
            if doc_id not in doc_data:
                doc_data[doc_id] = {
                    "text": docs[doc_idx]["text"],
                    "metadata": docs[doc_idx].get("metadata", {}),
                    "sparse_rank": rank + 1,
                }
            else:
                doc_data[doc_id]["sparse_rank"] = rank + 1
        
        # Sort by RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            data = doc_data[doc_id]
            results.append(HybridResult(
                id=doc_id,
                text=data["text"],
                score=score,
                metadata=data["metadata"],
                dense_rank=data.get("dense_rank"),
                sparse_rank=data.get("sparse_rank"),
            ))
        
        return results


class QueryExpander:
    """
    Query expansion and transformation.
    
    Improves retrieval by reformulating or expanding queries.
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize query expander.
        
        Args:
            llm_client: Optional LLM for query expansion
        """
        self.llm_client = llm_client
    
    def expand(self, query: str) -> list[str]:
        """
        Expand a query into multiple related queries.
        
        Args:
            query: Original query
            
        Returns:
            List of expanded queries including original
        """
        queries = [query]
        
        if self.llm_client:
            try:
                expanded = self._llm_expand(query)
                queries.extend(expanded)
            except Exception as e:
                print(f"Query expansion failed: {e}")
        
        return queries
    
    def _llm_expand(self, query: str) -> list[str]:
        """Use LLM to generate query expansions."""
        prompt = f"""Generate 2 alternative search queries for the following query. 
The alternatives should capture similar intent but use different words.

Original query: {query}

Return only the alternative queries, one per line."""

        response = self.llm_client.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        return [
            q.strip()
            for q in content.strip().split('\n')
            if q.strip() and q.strip() != query
        ][:2]
    
    def hyde(self, query: str) -> str:
        """
        Hypothetical Document Embeddings (HyDE).
        
        Generate a hypothetical answer to embed instead of the query.
        
        Args:
            query: Original query
            
        Returns:
            Hypothetical document text
        """
        if not self.llm_client:
            return query
        
        prompt = f"""Write a short paragraph that would be a good answer to this question:

Question: {query}

Write a direct, factual response (about 2-3 sentences)."""

        try:
            response = self.llm_client.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            return content.strip()
        except Exception:
            return query
