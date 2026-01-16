"""
Vector store wrapper for Qdrant.

Provides persistent storage with support for:
- Dense vector search
- Hybrid search (dense + sparse)
- Metadata filtering
"""

from typing import Optional, Union
from pathlib import Path
from dataclasses import dataclass
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    ScoredPoint,
)
import numpy as np


@dataclass
class SearchResult:
    """A single search result."""
    id: str
    text: str
    score: float
    metadata: dict


class QdrantStore:
    """
    Qdrant vector store wrapper with persistent storage.
    
    Supports both in-memory and file-based persistence.
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        path: Optional[str] = None,
        embedding_dim: int = 1024,
        distance: str = "cosine",
    ):
        """
        Initialize Qdrant store.
        
        Args:
            collection_name: Name of the collection
            path: Path for persistent storage (None for in-memory)
            embedding_dim: Dimension of embeddings
            distance: Distance metric (cosine, euclidean, dot)
        """
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        # Distance mapping
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        self.distance = distance_map.get(distance, Distance.COSINE)
        
        # Initialize client
        if path:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
            self.persistent = True
        else:
            self.client = QdrantClient(":memory:")
            self.persistent = False
        
        # Create or verify collection
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=self.distance,
                ),
            )
    
    def add(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Add documents to the store.
        
        Args:
            texts: Document texts
            embeddings: Document embeddings [n_docs, dim]
            metadatas: Optional metadata for each document
            ids: Optional IDs (generated if not provided)
            
        Returns:
            List of document IDs
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        points = []
        for i, (text, embedding, metadata, doc_id) in enumerate(
            zip(texts, embeddings, metadatas, ids)
        ):
            payload = {
                "text": text,
                **metadata,
            }
            
            points.append(PointStruct(
                id=doc_id if isinstance(doc_id, int) else i,
                vector=embedding.tolist(),
                payload=payload,
            ))
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        
        return ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_dict: Optional[dict] = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding [dim]
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of SearchResult objects
        """
        # Build filter if provided
        qdrant_filter = None
        if filter_dict:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_dict.items()
            ]
            qdrant_filter = Filter(must=conditions)
        
        # Use query_points (new API) instead of search
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=top_k,
            query_filter=qdrant_filter,
        )
        
        return [
            SearchResult(
                id=str(r.id),
                text=r.payload.get("text", "") if r.payload else "",
                score=r.score,
                metadata={k: v for k, v in (r.payload or {}).items() if k != "text"},
            )
            for r in results.points
        ]

    
    def get(self, doc_id: Union[str, int]) -> Optional[dict]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document payload or None
        """
        try:
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[doc_id if isinstance(doc_id, int) else int(doc_id)],
            )
            if results:
                return results[0].payload
        except Exception:
            pass
        return None
    
    def delete(self, doc_ids: list[Union[str, int]]):
        """
        Delete documents by ID.
        
        Args:
            doc_ids: List of document IDs to delete
        """
        int_ids = [int(i) if isinstance(i, str) else i for i in doc_ids]
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=int_ids,
        )
    
    def count(self) -> int:
        """Get the number of documents in the collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count
    
    def clear(self):
        """Delete all documents in the collection."""
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()


class BM25Store:
    """
    BM25 sparse retrieval store using bm25s.
    
    Provides keyword-based retrieval to complement dense search.
    """
    
    def __init__(
        self,
        index_path: Optional[str] = None,
    ):
        """
        Initialize BM25 store.
        
        Args:
            index_path: Path to save/load index
        """
        import bm25s
        import Stemmer
        
        self.index_path = index_path
        self.bm25 = bm25s.BM25()
        self.stemmer = Stemmer.Stemmer("english")
        self.documents = []
        self.is_indexed = False
    
    def index(self, documents: list[dict]):
        """
        Index documents for BM25 retrieval.
        
        Args:
            documents: List of dicts with 'text' key
        """
        import bm25s
        
        self.documents = documents
        texts = [doc["text"] for doc in documents]
        
        # Tokenize corpus - bm25s.tokenize returns a Tokenized object for lists
        corpus_tokens = bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer)
        
        # Index
        self.bm25.index(corpus_tokens)
        self.is_indexed = True
        
        # Save if path provided
        if self.index_path:
            self.save()

    
    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Search for documents matching query.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (doc_index, score) tuples
        """
        import bm25s
        
        if not self.is_indexed:
            return []
        
        # Tokenize query
        query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=self.stemmer)
        
        # Search
        results, scores = self.bm25.retrieve(query_tokens, k=top_k)
        
        # Flatten results (bm25s returns 2D arrays)
        doc_indices = results.flatten().tolist()
        doc_scores = scores.flatten().tolist()
        
        return list(zip(doc_indices, doc_scores))
    
    def save(self):
        """Save index to disk."""
        if self.index_path:
            Path(self.index_path).mkdir(parents=True, exist_ok=True)
            self.bm25.save(self.index_path, corpus=self.documents)
    
    def load(self):
        """Load index from disk."""
        if self.index_path and Path(self.index_path).exists():
            import bm25s
            self.bm25 = bm25s.BM25.load(self.index_path, load_corpus=True)
            self.documents = self.bm25.corpus
            self.is_indexed = True
