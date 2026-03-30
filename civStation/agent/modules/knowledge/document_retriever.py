"""
Document Retriever - Local document retrieval with optional embeddings.

Retrieves knowledge from local documents like Civopedia entries,
strategy guides, and game manuals. Supports both embedding-based
semantic search and keyword-based fallback.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from civStation.agent.modules.knowledge.base_retriever import BaseKnowledgeRetriever
from civStation.agent.modules.knowledge.schemas.knowledge_schemas import (
    KnowledgeChunk,
    KnowledgeSource,
)

logger = logging.getLogger(__name__)


class DocumentRetriever(BaseKnowledgeRetriever):
    """
    Retriever for local document collections.

    Supports:
    - Embedding-based semantic search (if embedding provider available)
    - Keyword/BM25-style fallback search
    - JSON-based document index
    """

    def __init__(
        self,
        index_path: Path | str | None = None,
        embedding_provider: Any = None,
    ):
        """
        Initialize the document retriever.

        Args:
            index_path: Path to the document index JSON file
            embedding_provider: Optional embedding provider for semantic search
        """
        self.index_path = Path(index_path) if index_path else None
        self.embedding_provider = embedding_provider

        # Document store: {doc_id: {"content": ..., "title": ..., "source": ..., "embedding": ...}}
        self.documents: dict[str, dict[str, Any]] = {}

        # Load existing index if provided
        if self.index_path and self.index_path.exists():
            self._load_index()

    def _load_index(self) -> None:
        """Load document index from file."""
        try:
            with open(self.index_path, encoding="utf-8") as f:
                data = json.load(f)
                self.documents = data.get("documents", {})
                logger.info(f"Loaded {len(self.documents)} documents from {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to load document index: {e}")
            self.documents = {}

    def save_index(self) -> None:
        """Save document index to file."""
        if not self.index_path:
            logger.warning("No index path specified, cannot save")
            return

        try:
            # Create parent directory if needed
            self.index_path.parent.mkdir(parents=True, exist_ok=True)

            # Don't save embeddings to JSON (too large)
            save_data = {
                "documents": {
                    doc_id: {k: v for k, v in doc.items() if k != "embedding"} for doc_id, doc in self.documents.items()
                }
            }

            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved {len(self.documents)} documents to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save document index: {e}")

    def add_document(
        self,
        doc_id: str,
        content: str,
        title: str,
        source: KnowledgeSource = KnowledgeSource.CUSTOM,
        metadata: dict | None = None,
    ) -> None:
        """
        Add a document to the index.

        Args:
            doc_id: Unique identifier for the document
            content: Document text content
            title: Document title
            source: Knowledge source type
            metadata: Optional additional metadata
        """
        doc = {
            "content": content,
            "title": title,
            "source": source.value,
            "metadata": metadata or {},
        }

        # Generate embedding if provider available
        if self.embedding_provider:
            try:
                embedding = self._generate_embedding(content)
                doc["embedding"] = embedding
            except Exception as e:
                logger.warning(f"Failed to generate embedding for {doc_id}: {e}")

        self.documents[doc_id] = doc
        logger.debug(f"Added document: {doc_id}")

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the index.

        Args:
            doc_id: Document ID to remove

        Returns:
            True if removed, False if not found
        """
        if doc_id in self.documents:
            del self.documents[doc_id]
            return True
        return False

    def retrieve(self, query: str, top_k: int = 5) -> list[KnowledgeChunk]:
        """
        Retrieve relevant documents for a query.

        Uses embedding-based search if available, otherwise falls back
        to keyword matching.

        Args:
            query: Search query
            top_k: Maximum results to return

        Returns:
            List of KnowledgeChunk objects
        """
        if not self.documents:
            return []

        # Try embedding-based search first
        if self.embedding_provider and self._has_embeddings():
            return self._semantic_search(query, top_k)

        # Fall back to keyword search
        return self._keyword_search(query, top_k)

    def _semantic_search(self, query: str, top_k: int) -> list[KnowledgeChunk]:
        """Semantic search using embeddings."""
        try:
            query_embedding = self._generate_embedding(query)

            # Calculate cosine similarity with all documents
            scored_docs = []
            for doc_id, doc in self.documents.items():
                if "embedding" not in doc:
                    continue

                score = self._cosine_similarity(query_embedding, doc["embedding"])
                scored_docs.append((doc_id, doc, score))

            # Sort by score and take top_k
            scored_docs.sort(key=lambda x: x[2], reverse=True)

            chunks = []
            for _doc_id, doc, score in scored_docs[:top_k]:
                chunk = KnowledgeChunk(
                    content=doc["content"],
                    source=KnowledgeSource(doc.get("source", "custom")),
                    title=doc["title"],
                    relevance_score=score,
                    metadata=doc.get("metadata", {}),
                )
                chunks.append(chunk)

            return chunks
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int) -> list[KnowledgeChunk]:
        """Simple keyword-based search (BM25-like scoring)."""
        # Tokenize query
        query_terms = self._tokenize(query)

        if not query_terms:
            return []

        # Score each document
        scored_docs = []
        for doc_id, doc in self.documents.items():
            content = doc["content"].lower()
            title = doc["title"].lower()

            # Simple TF-IDF-like scoring
            score = 0.0
            for term in query_terms:
                # Count occurrences in content and title
                content_count = content.count(term)
                title_count = title.count(term)

                # Weight title matches higher
                score += content_count * 1.0 + title_count * 3.0

            if score > 0:
                # Normalize by document length
                score = score / (len(content.split()) + 1)
                scored_docs.append((doc_id, doc, score))

        # Sort by score and take top_k
        scored_docs.sort(key=lambda x: x[2], reverse=True)

        # Normalize scores to 0-1 range
        if scored_docs:
            max_score = scored_docs[0][2]
            min_score = scored_docs[-1][2] if len(scored_docs) > 1 else 0

        chunks = []
        for _doc_id, doc, score in scored_docs[:top_k]:
            # Normalize score
            if max_score > min_score:
                normalized_score = (score - min_score) / (max_score - min_score)
            else:
                normalized_score = 1.0 if score > 0 else 0.0

            chunk = KnowledgeChunk(
                content=doc["content"],
                source=KnowledgeSource(doc.get("source", "custom")),
                title=doc["title"],
                relevance_score=normalized_score,
                metadata=doc.get("metadata", {}),
            )
            chunks.append(chunk)

        return chunks

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for keyword search."""
        # Convert to lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r"[\w가-힣]+", text)
        # Filter short tokens
        return [t for t in tokens if len(t) > 1]

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text using the embedding provider."""
        if not self.embedding_provider:
            raise ValueError("No embedding provider available")

        # This should be implemented based on your embedding provider
        # Example: return self.embedding_provider.embed(text)
        raise NotImplementedError("Embedding generation not implemented")

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=True))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _has_embeddings(self) -> bool:
        """Check if any documents have embeddings."""
        return any("embedding" in doc for doc in self.documents.values())

    def is_available(self) -> bool:
        """Check if the retriever is available."""
        return len(self.documents) > 0

    def get_document_count(self) -> int:
        """Get the number of documents in the index."""
        return len(self.documents)

    def get_all_titles(self) -> list[str]:
        """Get all document titles."""
        return [doc["title"] for doc in self.documents.values()]
