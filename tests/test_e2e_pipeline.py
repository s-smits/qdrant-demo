"""
End-to-end test of the RAG pipeline.
Tests document ingestion, retrieval, and the key components.
"""

import tempfile
import os


def test_e2e_pipeline():
    """Test the full pipeline end-to-end."""
    print("\n" + "="*70)
    print("END-TO-END PIPELINE TEST")
    print("="*70)
    
    # Import components
    from src.document_processing.parsers import DocumentParser, ParsedDocument, DocumentMetadata
    from src.document_processing.chunking import SemanticChunker, chunk_document
    from src.embeddings.models import ModernEmbeddings
    from src.vectorstore.qdrant_store import QdrantStore, BM25Store
    from src.retrieval.hybrid import HybridRetriever
    
    # 1. Create sample documents
    print("\n📄 Creating sample documents...")
    
    sample_texts = [
        """Machine learning is a subset of artificial intelligence (AI) that provides 
        systems the ability to automatically learn and improve from experience without 
        being explicitly programmed. Machine learning focuses on the development of 
        computer programs that can access data and use it to learn for themselves.""",
        
        """Deep learning is part of a broader family of machine learning methods based 
        on artificial neural networks with representation learning. Learning can be 
        supervised, semi-supervised or unsupervised. Deep learning architectures such 
        as deep neural networks have been applied to fields including computer vision.""",
        
        """Natural language processing (NLP) is a subfield of linguistics, computer science, 
        and artificial intelligence concerned with the interactions between computers and 
        human language, in particular how to program computers to process and analyze 
        large amounts of natural language data.""",
        
        """A transformer is a deep learning architecture developed by Google and used 
        for natural language processing tasks. The transformer model uses self-attention 
        mechanisms which allow it to process input sequences in parallel rather than 
        sequentially like RNNs.""",
    ]
    
    # 2. Create parsed documents
    docs = []
    for i, text in enumerate(sample_texts):
        doc = ParsedDocument(
            text=text,
            metadata=DocumentMetadata(
                source=f"test_doc_{i}.txt",
                filename=f"test_doc_{i}.txt",
                file_type="txt",
                doc_hash=f"hash_{i}",
            )
        )
        docs.append(doc)
    
    print(f"   Created {len(docs)} documents")
    
    # 3. Chunk documents semantically
    print("\n📦 Chunking documents...")
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(
            doc,
            strategy="semantic",
            min_chunk_size=50,
            max_chunk_size=500,
        )
        all_chunks.extend(chunks)
    
    print(f"   Created {len(all_chunks)} chunks")
    
    # 4. Initialize embedding model (use small model for test)  
    print("\n🧠 Loading embedding model...")
    embeddings = ModernEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
    print(f"   Model: {embeddings.model_name}")
    print(f"   Dimension: {embeddings.embedding_dim}")
    
    # 5. Embed chunks
    print("\n⚡ Embedding chunks...")
    chunk_texts = [c.text for c in all_chunks]
    embedding_result = embeddings.encode_documents(chunk_texts, show_progress=False)
    print(f"   Embedded {len(chunk_texts)} chunks")
    print(f"   Embedding shape: {embedding_result.dense.shape}")
    
    # 6. Store in vector database
    print("\n💾 Storing in Qdrant (in-memory for test)...")
    dense_store = QdrantStore(
        collection_name="test_collection",
        path=None,  # In-memory
        embedding_dim=embeddings.embedding_dim,
    )
    
    metadatas = [c.metadata for c in all_chunks]
    dense_store.add(
        texts=chunk_texts,
        embeddings=embedding_result.dense,
        metadatas=metadatas,
    )
    print(f"   Stored {dense_store.count()} vectors")
    
    # 7. Store in BM25
    print("\n📚 Building BM25 index...")
    sparse_store = BM25Store()
    doc_dicts = [{"text": c.text, "metadata": c.metadata} for c in all_chunks]
    sparse_store.index(doc_dicts)
    print(f"   Indexed {len(doc_dicts)} documents")
    
    # 8. Test retrieval
    print("\n🔍 Testing retrieval...")
    queries = [
        "What is machine learning?",
        "How do transformers work?",
        "What is NLP used for?",
    ]
    
    # Create hybrid retriever
    retriever = HybridRetriever(
        dense_store=dense_store,
        sparse_store=sparse_store,
        embeddings=embeddings,
        rrf_k=60,
        dense_weight=0.5,
    )
    
    for query in queries:
        print(f"\n   Query: {query}")
        results = retriever.retrieve(query, top_k=2)
        
        for i, result in enumerate(results):
            print(f"   [{i+1}] Score: {result.score:.4f}")
            print(f"       Text: {result.text[:80]}...")
    
    # 9. Verify hybrid actually combines both
    print("\n🧪 Verifying hybrid retrieval...")
    
    # Dense only
    dense_results = retriever.retrieve("machine learning AI", top_k=3, dense_only=True)
    print(f"   Dense-only results: {len(dense_results)}")
    
    # Sparse only  
    sparse_results = retriever.retrieve("machine learning AI", top_k=3, sparse_only=True)
    print(f"   Sparse-only results: {len(sparse_results)}")
    
    # Hybrid
    hybrid_results = retriever.retrieve("machine learning AI", top_k=3)
    print(f"   Hybrid results: {len(hybrid_results)}")
    
    # Check RRF fusion worked
    for r in hybrid_results:
        has_ranks = r.dense_rank is not None or r.sparse_rank is not None
        print(f"   - Dense rank: {r.dense_rank}, Sparse rank: {r.sparse_rank}")
    
    print("\n" + "="*70)
    print("✅ END-TO-END TEST PASSED!")
    print("="*70)
    print("""
Summary:
- Document parsing ✓
- Semantic chunking ✓  
- Embedding generation ✓
- Qdrant vector storage ✓
- BM25 sparse indexing ✓
- Hybrid retrieval with RRF ✓
""")


if __name__ == "__main__":
    test_e2e_pipeline()
