"""
Semantic chunking strategies for document processing.

Implements multiple chunking approaches:
- Semantic: Splits at semantic boundaries using embedding similarity
- Fixed: Traditional fixed-size chunking with overlap
- Recursive: Character-based recursive splitting
"""

from typing import Optional
from dataclasses import dataclass, field
import re

import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .parsers import ParsedDocument, DocumentMetadata

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


@dataclass
class Chunk:
    """A chunk of text with metadata."""
    text: str
    metadata: dict = field(default_factory=dict)
    
    @property
    def char_count(self) -> int:
        return len(self.text)
    
    @property
    def word_count(self) -> int:
        return len(self.text.split())


class SemanticChunker:
    """
    Chunk documents at semantic boundaries.
    
    Uses sentence embeddings to detect topic shifts and creates
    chunks that preserve semantic coherence.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        device: str = "cpu",
    ):
        """
        Initialize semantic chunker.
        
        Args:
            model_name: Sentence transformer model for similarity computation
            similarity_threshold: Below this similarity, create a new chunk
            min_chunk_size: Minimum characters before allowing a split
            max_chunk_size: Force split if chunk exceeds this size
            device: Device for embedding computation
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def chunk(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        """
        Split text into semantically coherent chunks.
        
        Args:
            text: Text to chunk
            metadata: Base metadata to include in each chunk
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 1:
            return [Chunk(
                text=text.strip(),
                metadata={**(metadata or {}), "chunk_idx": 0, "strategy": "semantic"}
            )]
        
        # Embed all sentences
        embeddings = self.model.encode(sentences, show_progress_bar=False)
        
        # Find semantic boundaries and create chunks
        chunks = []
        current_sentences = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # Compute similarity between consecutive sentences
            sim = cosine_similarity(
                [embeddings[i - 1]],
                [embeddings[i]]
            )[0][0]
            
            current_text = " ".join(current_sentences)
            
            # Decide whether to split
            should_split = (
                # Semantic shift detected
                (sim < self.similarity_threshold and 
                 len(current_text) >= self.min_chunk_size)
                or
                # Max size exceeded
                len(current_text) >= self.max_chunk_size
            )
            
            if should_split:
                chunks.append(Chunk(
                    text=current_text.strip(),
                    metadata={
                        **(metadata or {}),
                        "chunk_idx": len(chunks),
                        "strategy": "semantic",
                    }
                ))
                current_sentences = [sentences[i]]
            else:
                current_sentences.append(sentences[i])
        
        # Don't forget the last chunk
        if current_sentences:
            chunks.append(Chunk(
                text=" ".join(current_sentences).strip(),
                metadata={
                    **(metadata or {}),
                    "chunk_idx": len(chunks),
                    "strategy": "semantic",
                }
            ))
        
        return chunks
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using NLTK."""
        # Clean up text
        text = re.sub(r'\s+', ' ', text).strip()
        
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            # Fallback to simple splitting
            sentences = re.split(r'[.!?]+\s+', text)
        
        # Filter empty sentences
        return [s.strip() for s in sentences if s.strip()]


class FixedChunker:
    """
    Traditional fixed-size chunking with overlap.
    
    Simple but effective baseline chunking strategy.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """
        Initialize fixed chunker.
        
        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        """
        Split text into fixed-size chunks with overlap.
        
        Args:
            text: Text to chunk
            metadata: Base metadata to include in each chunk
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []
        
        text = text.strip()
        
        if len(text) <= self.chunk_size:
            return [Chunk(
                text=text,
                metadata={**(metadata or {}), "chunk_idx": 0, "strategy": "fixed"}
            )]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at a sentence or word boundary
            if end < len(text):
                # Look for sentence boundary
                boundary = self._find_boundary(text, end)
                if boundary > start:
                    end = boundary
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={
                        **(metadata or {}),
                        "chunk_idx": len(chunks),
                        "strategy": "fixed",
                    }
                ))
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _find_boundary(self, text: str, pos: int) -> int:
        """Find a good boundary near the given position."""
        # Look for sentence end (. ! ?)
        for i in range(pos, max(pos - 100, 0), -1):
            if text[i] in '.!?' and i + 1 < len(text) and text[i + 1] == ' ':
                return i + 2
        
        # Look for word boundary
        for i in range(pos, max(pos - 50, 0), -1):
            if text[i] == ' ':
                return i + 1
        
        return pos


class RecursiveChunker:
    """
    Recursive character-based chunking.
    
    Splits by a hierarchy of separators to preserve structure.
    """
    
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[list[str]] = None,
    ):
        """
        Initialize recursive chunker.
        
        Args:
            chunk_size: Target size for each chunk
            chunk_overlap: Overlap between chunks
            separators: Hierarchy of separators to split on
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
    
    def chunk(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        """
        Recursively split text into chunks.
        
        Args:
            text: Text to chunk
            metadata: Base metadata to include in each chunk
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []
        
        raw_chunks = self._split_recursive(text, self.separators)
        
        # Merge small chunks and split large ones
        merged = self._merge_splits(raw_chunks)
        
        return [
            Chunk(
                text=c.strip(),
                metadata={
                    **(metadata or {}),
                    "chunk_idx": i,
                    "strategy": "recursive",
                }
            )
            for i, c in enumerate(merged)
            if c.strip()
        ]
    
    def _split_recursive(
        self,
        text: str,
        separators: list[str],
    ) -> list[str]:
        """Recursively split text using separator hierarchy."""
        if not separators:
            return [text]
        
        separator = separators[0]
        remaining_seps = separators[1:]
        
        if separator == "":
            # Character-level split
            return list(text)
        
        splits = text.split(separator)
        
        result = []
        for split in splits:
            if len(split) > self.chunk_size and remaining_seps:
                # Recursively split larger chunks
                result.extend(self._split_recursive(split, remaining_seps))
            else:
                result.append(split)
        
        return result
    
    def _merge_splits(self, splits: list[str]) -> list[str]:
        """Merge small splits and ensure proper overlap."""
        merged = []
        current = ""
        
        for split in splits:
            if len(current) + len(split) + 1 <= self.chunk_size:
                current = f"{current} {split}".strip() if current else split
            else:
                if current:
                    merged.append(current)
                current = split
        
        if current:
            merged.append(current)
        
        return merged


def chunk_document(
    document: ParsedDocument,
    strategy: str = "semantic",
    **kwargs,
) -> list[Chunk]:
    """
    Chunk a parsed document using the specified strategy.
    
    Args:
        document: ParsedDocument to chunk
        strategy: One of "semantic", "fixed", "recursive"
        **kwargs: Arguments passed to the chunker
        
    Returns:
        List of Chunk objects
    """
    # Prepare base metadata
    base_metadata = {
        "source": document.metadata.source,
        "filename": document.metadata.filename,
        "doc_hash": document.metadata.doc_hash,
    }
    
    if strategy == "semantic":
        chunker = SemanticChunker(**kwargs)
    elif strategy == "fixed":
        chunker = FixedChunker(**kwargs)
    elif strategy == "recursive":
        chunker = RecursiveChunker(**kwargs)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    return chunker.chunk(document.text, metadata=base_metadata)


def chunk_documents(
    documents: list[ParsedDocument],
    strategy: str = "semantic",
    **kwargs,
) -> list[Chunk]:
    """
    Chunk multiple documents.
    
    Args:
        documents: List of ParsedDocument objects
        strategy: Chunking strategy to use
        **kwargs: Arguments passed to the chunker
        
    Returns:
        List of all chunks from all documents
    """
    all_chunks = []
    
    for doc in documents:
        chunks = chunk_document(doc, strategy=strategy, **kwargs)
        all_chunks.extend(chunks)
    
    return all_chunks
