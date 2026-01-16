"""
Modern embedding model wrappers.

Supports state-of-the-art models including:
- BGE-M3 (multi-vector: dense, sparse, ColBERT)
- Generic SentenceTransformer models
"""

from typing import Optional, Union
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class EmbeddingResult:
    """Result of embedding computation."""
    dense: np.ndarray  # Dense embeddings [n_texts, dim]
    sparse: Optional[dict] = None  # Sparse/lexical weights
    colbert: Optional[np.ndarray] = None  # ColBERT token embeddings


class ModernEmbeddings:
    """
    State-of-the-art embedding model wrapper.
    
    Supports BGE-M3 with multi-vector retrieval (dense + sparse + ColBERT)
    and standard SentenceTransformer models.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "auto",
        use_fp16: bool = True,
    ):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use (auto, cuda, mps, cpu)
            use_fp16: Use FP16 for faster inference
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
        self.use_fp16 = use_fp16 and device != "cpu"
        
        # Detect model type and load appropriately
        self._is_bge_m3 = "bge-m3" in model_name.lower()
        
        if self._is_bge_m3:
            self._load_bge_m3()
        else:
            self._load_sentence_transformer()
    
    def _load_bge_m3(self):
        """Load BGE-M3 model with FlagEmbedding."""
        try:
            from FlagEmbedding import BGEM3FlagModel
            
            self.model = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.use_fp16,
                device=self.device,
            )
            self.multi_vector = True
            self.embedding_dim = 1024  # BGE-M3 dimension
            
        except ImportError:
            print("FlagEmbedding not installed, falling back to SentenceTransformer")
            self._load_sentence_transformer()
    
    def _load_sentence_transformer(self):
        """Load standard SentenceTransformer model."""
        from sentence_transformers import SentenceTransformer
        
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.multi_vector = False
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def encode(
        self,
        texts: Union[str, list[str]],
        return_sparse: bool = False,
        return_colbert: bool = False,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> EmbeddingResult:
        """
        Encode texts to embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            return_sparse: Return sparse/lexical weights (BGE-M3 only)
            return_colbert: Return ColBERT token embeddings (BGE-M3 only)
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            EmbeddingResult with dense embeddings and optional sparse/ColBERT
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if self.multi_vector and self._is_bge_m3:
            result = self.model.encode(
                texts,
                batch_size=batch_size,
                return_dense=True,
                return_sparse=return_sparse,
                return_colbert_vecs=return_colbert,
            )
            
            return EmbeddingResult(
                dense=np.array(result["dense_vecs"]),
                sparse=result.get("lexical_weights"),
                colbert=result.get("colbert_vecs"),
            )
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
            
            return EmbeddingResult(dense=embeddings)
    
    def encode_query(
        self,
        query: str,
        return_sparse: bool = False,
    ) -> EmbeddingResult:
        """
        Encode a query with appropriate instruction prefix.
        
        Some models (like BGE) benefit from specific prefixes for queries.
        
        Args:
            query: Query text to encode
            return_sparse: Return sparse weights for hybrid search
            
        Returns:
            EmbeddingResult for the query
        """
        # Add instruction prefix for BGE models
        if "bge" in self.model_name.lower() and not self._is_bge_m3:
            query = f"Represent this sentence for searching relevant passages: {query}"
        
        return self.encode(
            [query],
            return_sparse=return_sparse,
        )
    
    def encode_documents(
        self,
        documents: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> EmbeddingResult:
        """
        Encode documents for indexing.
        
        Args:
            documents: List of document texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            EmbeddingResult with document embeddings
        """
        return self.encode(
            documents,
            batch_size=batch_size,
            show_progress=show_progress,
            return_sparse=self.multi_vector,  # Get sparse for hybrid search
        )
    
    def similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.
        
        Args:
            query_embedding: Query embedding [dim] or [1, dim]
            document_embeddings: Document embeddings [n_docs, dim]
            
        Returns:
            Similarity scores [n_docs]
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        doc_norm = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
        
        similarities = np.dot(query_norm, doc_norm.T).squeeze()
        
        return similarities


class EmbeddingCache:
    """
    Cache for computed embeddings.
    
    Useful for avoiding recomputation during development and testing.
    """
    
    def __init__(self, cache_dir: str = "./data/embedding_cache"):
        self.cache_dir = cache_dir
        self._cache = {}
    
    def get(self, text_hash: str) -> Optional[np.ndarray]:
        """Get cached embedding by text hash."""
        return self._cache.get(text_hash)
    
    def set(self, text_hash: str, embedding: np.ndarray):
        """Cache an embedding."""
        self._cache[text_hash] = embedding
    
    def has(self, text_hash: str) -> bool:
        """Check if embedding is cached."""
        return text_hash in self._cache
