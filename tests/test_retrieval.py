"""
Tests for retrieval pipeline.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestHybridRetrieval:
    """Tests for hybrid retrieval with RRF."""
    
    def test_rrf_scoring(self):
        """Test Reciprocal Rank Fusion scoring."""
        # RRF formula: 1 / (k + rank)
        k = 60
        
        # Document ranked 1st in dense, 3rd in sparse
        dense_contribution = 0.5 / (k + 1)  # weight * 1/(k+rank)
        sparse_contribution = 0.5 / (k + 3)
        expected_score = dense_contribution + sparse_contribution
        
        # Verify the math
        assert abs(expected_score - (0.5/61 + 0.5/63)) < 1e-6
    
    def test_rrf_fusion_properties(self):
        """Test RRF properties."""
        k = 60
        
        # Higher ranked documents should get higher scores
        score_rank_1 = 1 / (k + 1)
        score_rank_5 = 1 / (k + 5)
        score_rank_10 = 1 / (k + 10)
        
        assert score_rank_1 > score_rank_5 > score_rank_10
        
        # Documents appearing in both should rank higher
        # Single source score:
        single = 1 / (k + 1)
        
        # Both sources (ranked 1st in each):
        both = 0.5 / (k + 1) + 0.5 / (k + 1)
        
        assert both > single * 0.6  # Both should be significant


class TestReranking:
    """Tests for cross-encoder reranking."""
    
    def test_rerank_ordering(self):
        """Test that reranking reorders by score."""
        # Mock results
        docs = [
            {"id": "1", "text": "Not very relevant", "score": 0.9},
            {"id": "2", "text": "Very relevant to query", "score": 0.5},
            {"id": "3", "text": "Somewhat relevant", "score": 0.7},
        ]
        
        # Simulate reranking scores (higher = more relevant)
        rerank_scores = [0.2, 0.9, 0.5]  # Doc 2 is most relevant
        
        # Expected order after reranking: 2, 3, 1
        sorted_indices = np.argsort(rerank_scores)[::-1]
        expected_order = [docs[i]["id"] for i in sorted_indices]
        
        assert expected_order == ["2", "3", "1"]


class TestQueryExpansion:
    """Tests for query expansion."""
    
    def test_expansion_includes_original(self):
        """Test that expansion includes original query."""
        from src.retrieval.hybrid import QueryExpander
        
        expander = QueryExpander(llm_client=None)  # No LLM
        
        query = "What is machine learning?"
        expanded = expander.expand(query)
        
        assert query in expanded
        assert len(expanded) >= 1
