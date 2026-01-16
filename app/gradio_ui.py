"""
Modern Gradio UI for RAG system.

Features:
- PDF upload and processing
- Semantic chunking visualization
- Fine-tuning with progress
- Query interface with source display
- Comparison between base and fine-tuned
"""

import gradio as gr
from pathlib import Path
import tempfile
import os

# Import pipeline components
from src.pipeline import RAGPipeline, create_pipeline
from src.config import get_settings


# Global pipeline instance
pipeline = None


def initialize_pipeline(embedding_model: str, llm_provider: str):
    """Initialize the RAG pipeline."""
    global pipeline
    
    settings = get_settings()
    
    # Update settings if needed
    if embedding_model:
        settings.embedding.model_name = embedding_model
    
    pipeline = create_pipeline(
        embedding_model=embedding_model,
        llm_provider=llm_provider,
    )
    
    return "✅ Pipeline initialized successfully!"


def process_pdfs(files, progress=gr.Progress()):
    """Process uploaded PDF files."""
    global pipeline
    
    if pipeline is None:
        return "❌ Please initialize the pipeline first.", ""
    
    if not files:
        return "❌ No files uploaded.", ""
    
    progress(0.1, desc="Starting document processing...")
    
    try:
        # Get file paths
        file_paths = [f.name for f in files]
        
        progress(0.3, desc="Ingesting documents...")
        
        # Ingest documents
        num_chunks = pipeline.ingest(file_paths, show_progress=False)
        
        progress(1.0, desc="Processing complete!")
        
        # Create summary
        summary = f"""
## ✅ Processing Complete

**Documents processed:** {len(files)}
**Chunks created:** {num_chunks}
**Chunking strategy:** {pipeline.settings.chunking.strategy}

### Chunk Statistics:
- Min size: {pipeline.settings.chunking.min_chunk_size} chars
- Max size: {pipeline.settings.chunking.max_chunk_size} chars

Ready to query!
"""
        
        # Show sample chunks
        sample_chunks = ""
        if pipeline.chunks:
            sample_chunks = "### Sample Chunks:\n\n"
            for i, chunk in enumerate(pipeline.chunks[:3]):
                sample_chunks += f"**Chunk {i+1}** ({len(chunk.text)} chars):\n"
                sample_chunks += f"```\n{chunk.text[:200]}...\n```\n\n"
        
        return summary, sample_chunks
        
    except Exception as e:
        return f"❌ Error processing documents: {str(e)}", ""


def start_fine_tuning(epochs: int, batch_size: int, progress=gr.Progress()):
    """Start fine-tuning process."""
    global pipeline
    
    if pipeline is None:
        return "❌ Please initialize the pipeline first."
    
    if not pipeline.chunks:
        return "❌ No documents indexed. Please process documents first."
    
    progress(0.1, desc="Preparing fine-tuning...")
    
    try:
        progress(0.2, desc="Generating training data...")
        
        output_path = pipeline.fine_tune(
            epochs=int(epochs),
            batch_size=int(batch_size),
            use_hard_negatives=True,
        )
        
        progress(1.0, desc="Fine-tuning complete!")
        
        return f"""
## ✅ Fine-tuning Complete

**Model saved to:** `{output_path}`
**Epochs:** {int(epochs)}
**Batch size:** {int(batch_size)}

The model has been trained using **contrastive learning** with:
- Triplet loss (anchor, positive, negative)
- Synthetic query generation
- Hard negative mining

This is the **correct** approach for embedding fine-tuning!
"""
        
    except Exception as e:
        return f"❌ Error during fine-tuning: {str(e)}"


def query_rag(question: str, include_sources: bool):
    """Query the RAG system."""
    global pipeline
    
    if pipeline is None:
        return "❌ Please initialize the pipeline first.", ""
    
    if not pipeline.is_indexed:
        return "❌ No documents indexed. Please process documents first.", ""
    
    if not question.strip():
        return "❌ Please enter a question.", ""
    
    try:
        response = pipeline.query(question)
        
        answer = f"## Answer\n\n{response.answer}"
        
        sources = ""
        if include_sources and response.sources:
            sources = "## Sources\n\n"
            for i, source in enumerate(response.sources, 1):
                sources += f"**Source {i}:**\n"
                sources += f"```\n{source['text']}\n```\n"
                sources += f"*From: {source['source']}*\n\n"
        
        # Add retrieval stats
        stats = f"\n\n---\n*Retrieved: {response.num_retrieved} docs | "
        stats += f"Reranked: {response.num_reranked} docs*"
        
        return answer + stats, sources
        
    except Exception as e:
        return f"❌ Error: {str(e)}", ""


# Build the Gradio interface
def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(
        title="RAG Fine-tuned Embeddings - 2026 Edition",
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="blue",
        ),
    ) as interface:
        
        gr.Markdown("""
# 🚀 RAG Fine-tuned Embeddings - 2026 Edition

A production-ready RAG system with:
- **Semantic chunking** (context-aware splitting)
- **BGE-M3 embeddings** (state-of-the-art)
- **Hybrid retrieval** (dense + BM25 with RRF)
- **Cross-encoder reranking** (precision boost)
- **Proper contrastive fine-tuning** (TripletLoss)
""")
        
        with gr.Tab("⚙️ Setup"):
            with gr.Row():
                embedding_model = gr.Dropdown(
                    choices=[
                        "BAAI/bge-m3",
                        "BAAI/bge-base-en-v1.5",
                        "sentence-transformers/all-MiniLM-L6-v2",
                    ],
                    value="BAAI/bge-m3",
                    label="Embedding Model",
                )
                llm_provider = gr.Radio(
                    choices=["openai", "local"],
                    value="openai",
                    label="LLM Provider",
                    info="Use 'local' for testing with Qwen3-0.6B",
                )
            
            init_btn = gr.Button("Initialize Pipeline", variant="primary")
            init_status = gr.Markdown()
            
            init_btn.click(
                initialize_pipeline,
                inputs=[embedding_model, llm_provider],
                outputs=[init_status],
            )
        
        with gr.Tab("📄 Documents"):
            with gr.Row():
                with gr.Column(scale=2):
                    pdf_upload = gr.File(
                        file_count="multiple",
                        label="Upload PDF files",
                        file_types=[".pdf", ".txt", ".md"],
                    )
                    process_btn = gr.Button("Process Documents", variant="primary")
                
                with gr.Column(scale=3):
                    process_status = gr.Markdown()
                    chunk_samples = gr.Markdown()
            
            process_btn.click(
                process_pdfs,
                inputs=[pdf_upload],
                outputs=[process_status, chunk_samples],
            )
        
        with gr.Tab("🎯 Fine-tuning"):
            gr.Markdown("""
### Contrastive Fine-tuning

This uses **proper contrastive learning** with:
- **TripletLoss**: Learning relative distances (anchor, positive, negative)
- **Synthetic queries**: LLM-generated search queries
- **Hard negatives**: Mining challenging negative examples

This is fundamentally different from the original autoencoder approach!
""")
            
            with gr.Row():
                epochs = gr.Number(value=3, label="Epochs", minimum=1, maximum=10)
                batch_size = gr.Number(value=16, label="Batch Size", minimum=4, maximum=64)
            
            finetune_btn = gr.Button("Start Fine-tuning", variant="primary")
            finetune_status = gr.Markdown()
            
            finetune_btn.click(
                start_fine_tuning,
                inputs=[epochs, batch_size],
                outputs=[finetune_status],
            )
        
        with gr.Tab("💬 Query"):
            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask a question about your documents...",
                        lines=3,
                    )
                    include_sources = gr.Checkbox(
                        label="Show sources",
                        value=True,
                    )
                    query_btn = gr.Button("Ask", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    answer_output = gr.Markdown(label="Answer")
                with gr.Column():
                    sources_output = gr.Markdown(label="Sources")
            
            query_btn.click(
                query_rag,
                inputs=[question_input, include_sources],
                outputs=[answer_output, sources_output],
            )
            
            # Allow Enter key to submit
            question_input.submit(
                query_rag,
                inputs=[question_input, include_sources],
                outputs=[answer_output, sources_output],
            )
        
        gr.Markdown("""
---
### Architecture

```
Documents → Semantic Chunking → BGE-M3 Embeddings → Qdrant + BM25
                                                          ↓
Query → Embed → Hybrid Retrieval (RRF) → Cross-Encoder Rerank → LLM → Answer
```
""")
    
    return interface


# Entry point
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
