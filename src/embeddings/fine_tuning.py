"""
Proper contrastive fine-tuning for embedding models.

Unlike the original autoencoder approach, this implements:
- Triplet loss for learning relative distances
- InfoNCE/Multiple Negatives Ranking Loss
- Synthetic query generation for training data
- Hard negative mining
"""

from typing import Optional, Union
from pathlib import Path
import random
import json

import torch
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


class SyntheticQueryGenerator:
    """
    Generate synthetic queries for contrastive training.
    
    Uses an LLM to create realistic search queries that would
    retrieve a given document passage.
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize query generator.
        
        Args:
            llm_client: LangChain LLM or compatible client
        """
        self.llm_client = llm_client
    
    def generate(
        self,
        text: str,
        num_queries: int = 3,
    ) -> list[str]:
        """
        Generate synthetic queries for a text passage.
        
        Args:
            text: Document passage
            num_queries: Number of queries to generate
            
        Returns:
            List of generated queries
        """
        if self.llm_client is None:
            return self._generate_pseudo_queries(text, num_queries)
        
        return self._generate_llm_queries(text, num_queries)
    
    def _generate_llm_queries(
        self,
        text: str,
        num_queries: int,
    ) -> list[str]:
        """Generate queries using LLM."""
        prompt = f"""Generate {num_queries} diverse search queries that someone might use to find this text passage. The queries should:
1. Be natural questions or search terms
2. Cover different aspects of the content
3. Vary in specificity (some broad, some specific)

Text passage:
{text[:1000]}

Return only the queries, one per line, without numbering or bullet points."""

        try:
            response = self.llm_client.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            queries = [
                q.strip()
                for q in content.strip().split('\n')
                if q.strip() and len(q.strip()) > 5
            ]
            
            return queries[:num_queries]
            
        except Exception as e:
            print(f"LLM query generation failed: {e}, falling back to pseudo-queries")
            return self._generate_pseudo_queries(text, num_queries)
    
    def _generate_pseudo_queries(
        self,
        text: str,
        num_queries: int,
    ) -> list[str]:
        """
        Generate pseudo-queries without LLM.
        
        Uses extractive methods to create query-like text.
        """
        import nltk
        
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            sentences = text.split('. ')
        
        queries = []
        
        # Use first few sentences as pseudo-queries
        for sentence in sentences[:num_queries]:
            sentence = sentence.strip()
            if len(sentence) > 10:
                # Convert to question-like form (simple heuristic)
                if not sentence.endswith('?'):
                    words = sentence.split()
                    if len(words) > 3:
                        # Use key phrases
                        queries.append(' '.join(words[:min(10, len(words))]))
        
        # Pad with keyword extraction if needed
        while len(queries) < num_queries and sentences:
            # Extract key terms
            words = text.split()
            if len(words) > 5:
                start = random.randint(0, max(0, len(words) - 5))
                queries.append(' '.join(words[start:start + 5]))
        
        return queries[:num_queries]


class HardNegativeMiner:
    """
    Mine hard negatives for contrastive learning.
    
    Hard negatives are documents that are similar to the anchor
    but should NOT be retrieved for the query.
    """
    
    def __init__(self, embedding_model: SentenceTransformer):
        """
        Initialize miner with embedding model.
        
        Args:
            embedding_model: Model for computing similarities
        """
        self.model = embedding_model
    
    def mine(
        self,
        anchor_text: str,
        candidate_texts: list[str],
        positive_text: str,
        top_k: int = 5,
    ) -> list[str]:
        """
        Find hard negatives for a query-document pair.
        
        Args:
            anchor_text: The query or anchor text
            candidate_texts: Pool of potential negatives
            positive_text: The true positive document
            top_k: Number of hard negatives to return
            
        Returns:
            List of hard negative texts
        """
        if len(candidate_texts) <= 1:
            return []
        
        # Filter out the positive
        candidates = [t for t in candidate_texts if t != positive_text]
        
        if not candidates:
            return []
        
        # Embed anchor and candidates
        anchor_emb = self.model.encode([anchor_text])
        cand_embs = self.model.encode(candidates)
        
        # Compute similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(anchor_emb, cand_embs)[0]
        
        # Get top-k most similar (hardest negatives)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [candidates[i] for i in top_indices]


class ContrastiveFineTuner:
    """
    Fine-tune embedding models using proper contrastive learning.
    
    This is the correct approach, unlike the original autoencoder method.
    Uses:
    - TripletLoss or MultipleNegativesRankingLoss
    - Synthetic query generation
    - Hard negative mining
    """
    
    def __init__(
        self,
        base_model: str = "BAAI/bge-base-en-v1.5",
        device: str = "auto",
    ):
        """
        Initialize fine-tuner.
        
        Args:
            base_model: Base model to fine-tune
            device: Device for training
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.base_model = base_model
        self.model = SentenceTransformer(base_model, device=device)
        
        # Training metrics
        self.training_history = []
    
    def prepare_training_data(
        self,
        documents: list[dict],
        llm_client=None,
        queries_per_doc: int = 3,
        use_hard_negatives: bool = True,
        show_progress: bool = True,
    ) -> list[InputExample]:
        """
        Prepare training data for contrastive learning.
        
        Args:
            documents: List of dicts with 'text' key
            llm_client: Optional LLM for query generation
            queries_per_doc: Queries to generate per document
            use_hard_negatives: Mine hard negatives
            show_progress: Show progress bar
            
        Returns:
            List of InputExample for training
        """
        query_generator = SyntheticQueryGenerator(llm_client)
        hard_negative_miner = HardNegativeMiner(self.model) if use_hard_negatives else None
        
        # Get all document texts
        all_texts = [doc["text"] for doc in documents]
        
        examples = []
        iterator = tqdm(documents, desc="Preparing training data") if show_progress else documents
        
        for doc in iterator:
            text = doc["text"]
            
            # Generate synthetic queries
            queries = query_generator.generate(text, queries_per_doc)
            
            for query in queries:
                if use_hard_negatives and hard_negative_miner:
                    # Get hard negatives
                    negatives = hard_negative_miner.mine(
                        anchor_text=query,
                        candidate_texts=all_texts,
                        positive_text=text,
                        top_k=1,
                    )
                    negative = negatives[0] if negatives else self._random_negative(text, all_texts)
                else:
                    negative = self._random_negative(text, all_texts)
                
                # Create triplet example: (query, positive, negative)
                examples.append(InputExample(
                    texts=[query, text, negative]
                ))
        
        return examples
    
    def _random_negative(self, positive: str, all_texts: list[str]) -> str:
        """Get a random negative (not the positive)."""
        candidates = [t for t in all_texts if t != positive]
        return random.choice(candidates) if candidates else positive
    
    def fine_tune(
        self,
        train_examples: list[InputExample],
        epochs: int = 3,
        batch_size: int = 16,
        warmup_steps: int = 100,
        learning_rate: float = 2e-5,
        output_path: str = "./models/fine_tuned",
        loss_type: str = "triplet",
        evaluation_examples: Optional[list[InputExample]] = None,
    ) -> str:
        """
        Fine-tune the model with contrastive learning.
        
        Args:
            train_examples: Training examples from prepare_training_data
            epochs: Number of training epochs
            batch_size: Training batch size
            warmup_steps: Warmup steps for scheduler
            learning_rate: Learning rate
            output_path: Path to save fine-tuned model
            loss_type: "triplet" or "mnrl" (Multiple Negatives Ranking Loss)
            evaluation_examples: Optional examples for evaluation
            
        Returns:
            Path to saved model
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create data loader
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=batch_size,
        )
        
        # Select loss function
        if loss_type == "triplet":
            # Triplet loss: anchor, positive, negative
            train_loss = losses.TripletLoss(model=self.model)
        elif loss_type == "mnrl":
            # Multiple Negatives Ranking Loss (InfoNCE-style)
            # Works with pairs, uses in-batch negatives
            train_loss = losses.MultipleNegativesRankingLoss(model=self.model)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Prepare evaluator if examples provided
        evaluator = None
        if evaluation_examples:
            # Create simple evaluator
            evaluator = self._create_evaluator(evaluation_examples)
        
        print(f"\n{'='*60}")
        print(f"Starting contrastive fine-tuning")
        print(f"{'='*60}")
        print(f"Base model: {self.base_model}")
        print(f"Device: {self.device}")
        print(f"Training examples: {len(train_examples)}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"Loss function: {loss_type}")
        print(f"Output path: {output_path}")
        print(f"{'='*60}\n")
        
        # Train
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            optimizer_params={'lr': learning_rate},
            output_path=str(output_path),
            show_progress_bar=True,
            evaluator=evaluator,
            evaluation_steps=500 if evaluator else 0,
        )
        
        # Save training info
        training_info = {
            "base_model": self.base_model,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "loss_type": loss_type,
            "num_examples": len(train_examples),
        }
        
        with open(output_path / "training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        print(f"\n✅ Model saved to: {output_path}")
        
        return str(output_path)
    
    def _create_evaluator(self, examples: list[InputExample]):
        """Create a simple evaluator for training."""
        # Extract queries and positives for evaluation
        queries = {}
        corpus = {}
        relevant_docs = {}
        
        for i, ex in enumerate(examples):
            query_id = f"q{i}"
            doc_id = f"d{i}"
            
            queries[query_id] = ex.texts[0]
            corpus[doc_id] = ex.texts[1]
            relevant_docs[query_id] = {doc_id}
        
        return InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name="train_eval",
        )
    
    def load_fine_tuned(self, model_path: str) -> SentenceTransformer:
        """
        Load a fine-tuned model.
        
        Args:
            model_path: Path to fine-tuned model
            
        Returns:
            Loaded SentenceTransformer model
        """
        self.model = SentenceTransformer(model_path, device=self.device)
        return self.model
    
    def evaluate_improvement(
        self,
        test_queries: list[str],
        test_documents: list[str],
        relevant_indices: list[int],
    ) -> dict:
        """
        Evaluate retrieval improvement after fine-tuning.
        
        Args:
            test_queries: Test queries
            test_documents: Test document corpus
            relevant_indices: Index of relevant doc for each query
            
        Returns:
            Dict with evaluation metrics
        """
        query_embeddings = self.model.encode(test_queries)
        doc_embeddings = self.model.encode(test_documents)
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embeddings, doc_embeddings)
        
        # Compute metrics
        mrr = 0.0
        recall_at_5 = 0.0
        
        for i, rel_idx in enumerate(relevant_indices):
            rankings = np.argsort(similarities[i])[::-1]
            rank = np.where(rankings == rel_idx)[0][0] + 1
            
            mrr += 1.0 / rank
            if rank <= 5:
                recall_at_5 += 1.0
        
        n = len(test_queries)
        
        return {
            "mrr": mrr / n,
            "recall@5": recall_at_5 / n,
            "num_queries": n,
        }
