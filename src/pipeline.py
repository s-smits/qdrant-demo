"""
Main RAG pipeline orchestrating all components.

This is the primary entry point for the RAG system.
"""

from typing import Optional, Union
from pathlib import Path
from dataclasses import dataclass

from .config import Settings, get_settings
from .document_processing.parsers import DocumentParser, ParsedDocument, parse_documents
from .document_processing.chunking import chunk_documents, Chunk
from .embeddings.models import ModernEmbeddings
from .embeddings.fine_tuning import ContrastiveFineTuner
from .vectorstore.qdrant_store import QdrantStore, BM25Store
from .retrieval.hybrid import HybridRetriever
from .retrieval.reranker import CrossEncoderReranker
from .generation.llm import get_llm, RAGGenerator


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""
    answer: str
    sources: list[dict]
    query: str
    num_retrieved: int
    num_reranked: int


class RAGPipeline:
    """
    Complete RAG pipeline with modern 2026 best practices.
    
    Components:
    - Semantic chunking
    - BGE-M3 embeddings (or configurable)
    - Hybrid retrieval (dense + BM25 with RRF)
    - Cross-encoder reranking
    - LLM generation
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedding_model: Optional[str] = None,
        llm_provider: str = "openai",
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            settings: Configuration settings
            embedding_model: Override embedding model
            llm_provider: LLM provider ("openai" or "local")
        """
        self.settings = settings or get_settings()
        
        # Initialize components lazily
        self._parser = None
        self._embeddings = None
        self._dense_store = None
        self._sparse_store = None
        self._retriever = None
        self._reranker = None
        self._llm = None
        self._generator = None
        
        # Override settings if provided
        if embedding_model:
            self.settings.embedding.model_name = embedding_model
        
        self.llm_provider = llm_provider
        
        # Track indexed documents
        self.documents = []
        self.chunks = []
        self.is_indexed = False
    
    @property
    def parser(self) -> DocumentParser:
        if self._parser is None:
            self._parser = DocumentParser()
        return self._parser
    
    @property
    def embeddings(self) -> ModernEmbeddings:
        if self._embeddings is None:
            self._embeddings = ModernEmbeddings(
                model_name=self.settings.embedding.model_name,
                device=self.settings.embedding.device,
                use_fp16=self.settings.embedding.use_fp16,
            )
        return self._embeddings
    
    @property
    def dense_store(self) -> QdrantStore:
        if self._dense_store is None:
            self._dense_store = QdrantStore(
                collection_name=self.settings.vectorstore.collection_name,
                path=self.settings.vectorstore.qdrant_path,
                embedding_dim=self.embeddings.embedding_dim,
            )
        return self._dense_store
    
    @property
    def sparse_store(self) -> BM25Store:
        if self._sparse_store is None:
            self._sparse_store = BM25Store(
                index_path=self.settings.vectorstore.bm25_index_path,
            )
        return self._sparse_store
    
    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(
                dense_store=self.dense_store,
                sparse_store=self.sparse_store,
                embeddings=self.embeddings,
                rrf_k=self.settings.retrieval.rrf_k,
                dense_weight=self.settings.retrieval.hybrid_alpha,
            )
        return self._retriever
    
    @property
    def reranker(self) -> Optional[CrossEncoderReranker]:
        if self._reranker is None and self.settings.retrieval.use_reranker:
            self._reranker = CrossEncoderReranker(
                model_name=self.settings.embedding.reranker_model,
                device=self.settings.embedding.device,
            )
        return self._reranker
    
    @property
    def llm(self):
        if self._llm is None:
            if self.llm_provider == "openai":
                self._llm = get_llm(
                    provider="openai",
                    model_name=self.settings.llm.model_name,
                    api_key=self.settings.llm.openai_api_key,
                    temperature=self.settings.llm.temperature,
                )
            else:
                self._llm = get_llm(
                    provider="local",
                    model_name="Qwen/Qwen3-0.6B",  # Lightweight for testing
                )
        return self._llm
    
    @property
    def generator(self) -> RAGGenerator:
        if self._generator is None:
            self._generator = RAGGenerator(llm=self.llm)
        return self._generator
    
    def ingest(
        self,
        file_paths: list[Union[str, Path]],
        show_progress: bool = True,
    ) -> int:
        """
        Ingest and index documents.
        
        Args:
            file_paths: Paths to documents to ingest
            show_progress: Show progress bars
            
        Returns:
            Number of chunks indexed
        """
        print(f"\n{'='*60}")
        print("📄 Document Ingestion")
        print(f"{'='*60}")
        
        # Parse documents
        print(f"\nParsing {len(file_paths)} documents...")
        self.documents = parse_documents(file_paths)
        print(f"✅ Parsed {len(self.documents)} documents")
        
        # Chunk documents
        print(f"\nChunking with {self.settings.chunking.strategy} strategy...")
        self.chunks = chunk_documents(
            self.documents,
            strategy=self.settings.chunking.strategy,
            similarity_threshold=self.settings.chunking.similarity_threshold,
            min_chunk_size=self.settings.chunking.min_chunk_size,
            max_chunk_size=self.settings.chunking.max_chunk_size,
        )
        print(f"✅ Created {len(self.chunks)} chunks")
        
        # Embed chunks
        print(f"\nEmbedding chunks with {self.settings.embedding.model_name}...")
        chunk_texts = [c.text for c in self.chunks]
        embeddings_result = self.embeddings.encode_documents(
            chunk_texts,
            show_progress=show_progress,
        )
        print(f"✅ Generated embeddings (dim={self.embeddings.embedding_dim})")
        
        # Index in dense store
        print(f"\nIndexing in Qdrant...")
        metadatas = [c.metadata for c in self.chunks]
        self.dense_store.add(
            texts=chunk_texts,
            embeddings=embeddings_result.dense,
            metadatas=metadatas,
        )
        print(f"✅ Indexed {self.dense_store.count()} vectors")
        
        # Index in sparse store
        print(f"\nIndexing in BM25...")
        doc_dicts = [{"text": c.text, "metadata": c.metadata} for c in self.chunks]
        self.sparse_store.index(doc_dicts)
        print(f"✅ BM25 index created")
        
        self.is_indexed = True
        
        print(f"\n{'='*60}")
        print(f"✅ Ingestion complete: {len(self.chunks)} chunks indexed")
        print(f"{'='*60}\n")
        
        return len(self.chunks)
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        """
        Query the RAG system.
        
        Args:
            question: User question
            top_k: Number of documents to use for generation
            
        Returns:
            RAGResponse with answer and sources
        """
        if not self.is_indexed:
            return RAGResponse(
                answer="No documents indexed. Please ingest documents first.",
                sources=[],
                query=question,
                num_retrieved=0,
                num_reranked=0,
            )
        
        top_k = top_k or self.settings.retrieval.final_top_k
        initial_k = self.settings.retrieval.initial_top_k
        
        # Hybrid retrieval
        hybrid_results = self.retriever.retrieve(
            query=question,
            top_k=initial_k,
        )
        
        # Reranking
        if self.reranker and hybrid_results:
            reranked = self.reranker.rerank_hybrid_results(
                query=question,
                hybrid_results=hybrid_results,
                top_k=top_k,
            )
            
            context_docs = [
                {"text": r.text, "metadata": r.metadata, "score": r.rerank_score}
                for r in reranked
            ]
            num_reranked = len(reranked)
        else:
            context_docs = [
                {"text": r.text, "metadata": r.metadata, "score": r.score}
                for r in hybrid_results[:top_k]
            ]
            num_reranked = 0
        
        # Generate answer
        result = self.generator.generate_with_sources(
            question=question,
            context_docs=context_docs,
        )
        
        return RAGResponse(
            answer=result["answer"],
            sources=result["sources"],
            query=question,
            num_retrieved=len(hybrid_results),
            num_reranked=num_reranked,
        )
    
    def fine_tune(
        self,
        epochs: int = 3,
        batch_size: int = 16,
        use_hard_negatives: bool = True,
    ) -> str:
        """
        Fine-tune embeddings on ingested documents.
        
        Args:
            epochs: Training epochs
            batch_size: Training batch size
            use_hard_negatives: Mine hard negatives
            
        Returns:
            Path to fine-tuned model
        """
        if not self.chunks:
            raise ValueError("No documents ingested. Ingest documents first.")
        
        print(f"\n{'='*60}")
        print("🎯 Contrastive Fine-tuning")
        print(f"{'='*60}")
        
        # Use smaller base model for fine-tuning
        fine_tuner = ContrastiveFineTuner(
            base_model=self.settings.fine_tuning.base_model,
            device=self.settings.embedding.device,
        )
        
        # Prepare training data
        print("\nPreparing training data...")
        doc_dicts = [{"text": c.text, "metadata": c.metadata} for c in self.chunks]
        
        # Use LLM for query generation if available
        llm_client = self.llm if self.settings.fine_tuning.use_llm_for_queries else None
        
        train_examples = fine_tuner.prepare_training_data(
            documents=doc_dicts,
            llm_client=llm_client,
            queries_per_doc=self.settings.fine_tuning.queries_per_doc,
            use_hard_negatives=use_hard_negatives,
        )
        
        print(f"✅ Created {len(train_examples)} training examples")
        
        # Fine-tune
        output_path = fine_tuner.fine_tune(
            train_examples=train_examples,
            epochs=epochs,
            batch_size=batch_size,
            output_path=self.settings.fine_tuning.output_path,
        )
        
        return output_path


def create_pipeline(
    embedding_model: str = "BAAI/bge-m3",
    llm_provider: str = "openai",
) -> RAGPipeline:
    """
    Factory function to create a RAG pipeline.
    
    Args:
        embedding_model: Embedding model to use
        llm_provider: "openai" or "local"
        
    Returns:
        Configured RAGPipeline
    """
    return RAGPipeline(
        embedding_model=embedding_model,
        llm_provider=llm_provider,
    )
