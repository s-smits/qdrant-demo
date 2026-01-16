#!/usr/bin/env python3
"""
RAG Fine-tuned Embeddings

Main entry point for the application.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="RAG System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch UI
  python main.py ui
  
  # Ingest documents
  python main.py ingest --files doc1.pdf doc2.pdf
  
  # Query
  python main.py query "Search query"
  
  # Train
  python main.py finetune --epochs 3
""",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # UI command
    ui_parser = subparsers.add_parser("ui", help="Launch web interface")
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
        
        print("Launching interface at http://localhost:7860...")
        
        interface = create_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=args.share,
        )
    
    elif args.command == "ingest":
        from src.pipeline import create_pipeline
        
        print("Ingesting documents...")
        
        pipeline = create_pipeline(embedding_model=args.model)
        num_chunks = pipeline.ingest(args.files)
        
        print(f"Complete. Processed {len(args.files)} documents, {num_chunks} chunks.")
    
    elif args.command == "query":
        from src.pipeline import create_pipeline
        
        pipeline = create_pipeline(embedding_model=args.model)
        
        # Try to load existing index
        if pipeline.dense_store.count() == 0:
            print("Error: Index empty. Run 'ingest' first.")
            sys.exit(1)
        
        response = pipeline.query(args.question)
        
        print(f"\nQ: {args.question}")
        print(f"A: {response.answer}")
        print(f"\nStats: Retrieved {response.num_retrieved}, Reranked {response.num_reranked}")
    
    elif args.command == "finetune":
        from src.pipeline import create_pipeline
        
        print("Starting training (Contrastive Fine-tuning)...")
        
        pipeline = create_pipeline()
        
        if not pipeline.chunks:
            print("Error: Index empty. Run 'ingest' first.")
            sys.exit(1)
        
        output_path = pipeline.fine_tune(
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        
        print(f"Training complete. Model saved to: {output_path}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
