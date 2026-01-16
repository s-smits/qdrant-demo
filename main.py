#!/usr/bin/env python3
"""
RAG Fine-tuned Embeddings - 2026 Edition

Main entry point for the application.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="RAG Fine-tuned Embeddings - 2026 Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch Gradio UI
  python main.py ui
  
  # Process documents from command line
  python main.py ingest --files doc1.pdf doc2.pdf
  
  # Query the system
  python main.py query "What is the main topic?"
  
  # Fine-tune embeddings
  python main.py finetune --epochs 3
""",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # UI command
    ui_parser = subparsers.add_parser("ui", help="Launch Gradio web interface")
    ui_parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    ui_parser.add_argument("--share", action="store_true", help="Create public link")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("--files", nargs="+", required=True, help="Files to ingest")
    ingest_parser.add_argument("--model", default="BAAI/bge-m3", help="Embedding model")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--model", default="BAAI/bge-m3", help="Embedding model")
    
    # Fine-tune command
    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune embeddings")
    finetune_parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    finetune_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    
    args = parser.parse_args()
    
    if args.command == "ui":
        from app.gradio_ui import create_interface
        
        print("\n" + "="*60)
        print("🚀 RAG Fine-tuned Embeddings - 2026 Edition")
        print("="*60)
        print("\nLaunching Gradio interface...")
        
        interface = create_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=args.share,
        )
    
    elif args.command == "ingest":
        from src.pipeline import create_pipeline
        
        print("\n" + "="*60)
        print("📄 Document Ingestion")
        print("="*60)
        
        pipeline = create_pipeline(embedding_model=args.model)
        num_chunks = pipeline.ingest(args.files)
        
        print(f"\n✅ Ingested {len(args.files)} documents into {num_chunks} chunks")
    
    elif args.command == "query":
        from src.pipeline import create_pipeline
        
        print("\n" + "="*60)
        print("💬 RAG Query")
        print("="*60)
        
        pipeline = create_pipeline(embedding_model=args.model)
        
        # Try to load existing index
        if pipeline.dense_store.count() == 0:
            print("\n❌ No documents indexed. Run 'ingest' first.")
            sys.exit(1)
        
        response = pipeline.query(args.question)
        
        print(f"\n**Question:** {args.question}")
        print(f"\n**Answer:** {response.answer}")
        print(f"\n*Retrieved {response.num_retrieved} docs, reranked {response.num_reranked}*")
    
    elif args.command == "finetune":
        from src.pipeline import create_pipeline
        
        print("\n" + "="*60)
        print("🎯 Contrastive Fine-tuning")
        print("="*60)
        
        pipeline = create_pipeline()
        
        if not pipeline.chunks:
            print("\n❌ No documents indexed. Run 'ingest' first.")
            sys.exit(1)
        
        output_path = pipeline.fine_tune(
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        
        print(f"\n✅ Fine-tuned model saved to: {output_path}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
