"""
Document parsing utilities.
Handles extraction of text and metadata from various document formats.
"""

from pathlib import Path
from typing import Union, Optional
from dataclasses import dataclass, field
import hashlib
from datetime import datetime

from PyPDF2 import PdfReader


@dataclass
class DocumentMetadata:
    """Metadata extracted from a document."""
    source: str
    filename: str
    file_type: str
    page_count: Optional[int] = None
    title: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    doc_hash: Optional[str] = None
    extra: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """A parsed document with text and metadata."""
    text: str
    metadata: DocumentMetadata
    pages: Optional[list[str]] = None  # Text per page for PDFs


class DocumentParser:
    """
    Parse documents from various formats.
    
    Supports:
    - PDF files
    - Plain text files
    - (Extensible for more formats)
    """
    
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
    
    def parse(
        self,
        file_path: Union[str, Path],
        extract_pages: bool = True,
    ) -> ParsedDocument:
        """
        Parse a document and extract text with metadata.
        
        Args:
            file_path: Path to the document
            extract_pages: For PDFs, keep track of page-level text
            
        Returns:
            ParsedDocument with extracted text and metadata
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        
        extension = path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )
        
        if extension == ".pdf":
            return self._parse_pdf(path, extract_pages)
        elif extension in {".txt", ".md"}:
            return self._parse_text(path)
        else:
            raise ValueError(f"No parser implemented for: {extension}")
    
    def _parse_pdf(self, path: Path, extract_pages: bool) -> ParsedDocument:
        """Parse a PDF file."""
        reader = PdfReader(str(path))
        
        # Extract text from all pages
        pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)
        
        full_text = "\n\n".join(pages)
        
        # Extract metadata from PDF
        pdf_metadata = reader.metadata or {}
        
        metadata = DocumentMetadata(
            source=str(path.absolute()),
            filename=path.name,
            file_type="pdf",
            page_count=len(reader.pages),
            title=pdf_metadata.get("/Title"),
            author=pdf_metadata.get("/Author"),
            doc_hash=self._compute_hash(full_text),
        )
        
        return ParsedDocument(
            text=full_text,
            metadata=metadata,
            pages=pages if extract_pages else None,
        )
    
    def _parse_text(self, path: Path) -> ParsedDocument:
        """Parse a plain text or markdown file."""
        text = path.read_text(encoding="utf-8")
        
        metadata = DocumentMetadata(
            source=str(path.absolute()),
            filename=path.name,
            file_type=path.suffix.lstrip("."),
            doc_hash=self._compute_hash(text),
        )
        
        return ParsedDocument(
            text=text,
            metadata=metadata,
        )
    
    @staticmethod
    def _compute_hash(text: str) -> str:
        """Compute a hash of the document content for deduplication."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]


def parse_documents(
    paths: list[Union[str, Path]],
    skip_errors: bool = True,
) -> list[ParsedDocument]:
    """
    Parse multiple documents.
    
    Args:
        paths: List of file paths to parse
        skip_errors: If True, skip files that fail to parse
        
    Returns:
        List of parsed documents
    """
    parser = DocumentParser()
    documents = []
    
    for path in paths:
        try:
            doc = parser.parse(path)
            documents.append(doc)
        except Exception as e:
            if skip_errors:
                print(f"Warning: Failed to parse {path}: {e}")
            else:
                raise
    
    return documents
