"""
Configuration management for RAG system.
Uses pydantic-settings for validation and environment variable support.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal
from functools import lru_cache
import torch


def get_default_device() -> str:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class EmbeddingConfig(BaseSettings):
    """Configuration for embedding models."""
    
    # Primary embedding model for document/query encoding
    model_name: str = Field(
        default="BAAI/bge-m3",
        description="HuggingFace model name for embeddings"
    )
    
    # Lightweight model for semantic chunking boundary detection
    chunking_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Fast model for chunking similarity computation"
    )
    
    # Reranker model
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Cross-encoder model for reranking"
    )
    
    device: str = Field(
        default_factory=get_default_device,
        description="Device to use: cuda, mps, or cpu"
    )
    
    use_fp16: bool = Field(
        default=True,
        description="Use FP16 for faster inference (disabled on CPU)"
    )
    
    model_config = {"env_prefix": "EMBEDDING_"}


class ChunkingConfig(BaseSettings):
    """Configuration for document chunking."""
    
    strategy: Literal["semantic", "fixed", "recursive"] = Field(
        default="semantic",
        description="Chunking strategy to use"
    )
    
    # Semantic chunking parameters
    similarity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for semantic boundary detection"
    )
    
    min_chunk_size: int = Field(
        default=100,
        ge=50,
        description="Minimum chunk size in characters"
    )
    
    max_chunk_size: int = Field(
        default=1000,
        le=4000,
        description="Maximum chunk size in characters"
    )
    
    # Fixed/recursive chunking parameters
    chunk_size: int = Field(
        default=512,
        description="Target chunk size for fixed/recursive chunking"
    )
    
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between chunks"
    )
    
    model_config = {"env_prefix": "CHUNKING_"}


class RetrievalConfig(BaseSettings):
    """Configuration for retrieval pipeline."""
    
    # Hybrid retrieval
    hybrid_alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for dense vs sparse (1.0 = dense only, 0.0 = sparse only)"
    )
    
    rrf_k: int = Field(
        default=60,
        description="RRF constant for score fusion"
    )
    
    # Retrieval parameters
    initial_top_k: int = Field(
        default=20,
        description="Number of documents to retrieve before reranking"
    )
    
    final_top_k: int = Field(
        default=5,
        description="Number of documents after reranking"
    )
    
    # Reranking
    use_reranker: bool = Field(
        default=True,
        description="Whether to use cross-encoder reranking"
    )
    
    model_config = {"env_prefix": "RETRIEVAL_"}


class VectorStoreConfig(BaseSettings):
    """Configuration for vector storage."""
    
    provider: Literal["qdrant", "qdrant_memory"] = Field(
        default="qdrant",
        description="Vector store provider"
    )
    
    # Qdrant settings
    qdrant_path: str = Field(
        default="./data/qdrant",
        description="Path for Qdrant persistent storage"
    )
    
    collection_name: str = Field(
        default="documents",
        description="Name of the vector collection"
    )
    
    # BM25 settings
    bm25_index_path: str = Field(
        default="./data/bm25_index",
        description="Path for BM25 index storage"
    )
    
    model_config = {"env_prefix": "VECTORSTORE_"}


class LLMConfig(BaseSettings):
    """Configuration for LLM generation."""
    
    provider: Literal["openai", "anthropic"] = Field(
        default="openai",
        description="LLM provider"
    )
    
    model_name: str = Field(
        default="gpt-4o",
        description="Model name for generation"
    )
    
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    
    max_tokens: int = Field(
        default=1024,
        description="Maximum tokens in response"
    )
    
    # API keys (loaded from environment)
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key"
    )
    
    model_config = {"env_prefix": "LLM_", "env_file": ".env"}


class FineTuningConfig(BaseSettings):
    """Configuration for embedding fine-tuning."""
    
    enabled: bool = Field(
        default=True,
        description="Whether fine-tuning is enabled"
    )
    
    base_model: str = Field(
        default="BAAI/bge-base-en-v1.5",
        description="Base model for fine-tuning (smaller than production model)"
    )
    
    output_path: str = Field(
        default="./models/fine_tuned",
        description="Path to save fine-tuned model"
    )
    
    # Training parameters
    epochs: int = Field(
        default=3,
        ge=1,
        description="Number of training epochs"
    )
    
    batch_size: int = Field(
        default=16,
        ge=1,
        description="Training batch size"
    )
    
    learning_rate: float = Field(
        default=2e-5,
        description="Learning rate"
    )
    
    warmup_steps: int = Field(
        default=100,
        description="Warmup steps for scheduler"
    )
    
    # Synthetic data generation
    use_llm_for_queries: bool = Field(
        default=True,
        description="Use LLM to generate synthetic queries"
    )
    
    queries_per_doc: int = Field(
        default=3,
        description="Number of synthetic queries per document"
    )
    
    model_config = {"env_prefix": "FINETUNE_"}


class Settings(BaseSettings):
    """Main settings aggregating all config sections."""
    
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    fine_tuning: FineTuningConfig = Field(default_factory=FineTuningConfig)
    
    # Application settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience function for quick access
settings = get_settings()
