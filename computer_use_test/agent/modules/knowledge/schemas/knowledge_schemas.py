"""
Knowledge Schemas - Data structures for knowledge retrieval.

Defines the data structures used for representing knowledge chunks,
query results, and knowledge sources.
"""

from dataclasses import dataclass, field
from enum import Enum


class KnowledgeSource(str, Enum):
    """Sources of knowledge for the RAG system."""

    CIVOPEDIA = "civopedia"  # In-game encyclopedia
    WEB_SEARCH = "web_search"  # Web search results
    STRATEGY_GUIDE = "strategy_guide"  # External strategy guides
    GAME_MANUAL = "game_manual"  # Official game manual
    COMMUNITY_WIKI = "community_wiki"  # Community wikis (e.g., Civ Wiki)
    CUSTOM = "custom"  # Custom documents


@dataclass
class KnowledgeChunk:
    """
    A single chunk of knowledge retrieved from a source.

    Represents a piece of information that can be used to
    augment prompts or answer questions.
    """

    content: str  # The actual text content
    source: KnowledgeSource  # Where this came from
    title: str  # Title or heading of the content
    relevance_score: float = 0.0  # Similarity/relevance score (0-1)
    url: str = ""  # URL if from web source
    metadata: dict = field(default_factory=dict)  # Additional metadata

    def __str__(self) -> str:
        return f"[{self.source.value}] {self.title}: {self.content[:100]}..."

    def to_prompt_string(self) -> str:
        """Format as a string suitable for inclusion in prompts."""
        lines = [f"### {self.title}"]
        if self.url:
            lines.append(f"출처: {self.url}")
        lines.append(self.content)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "source": self.source.value,
            "title": self.title,
            "relevance_score": self.relevance_score,
            "url": self.url,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeChunk":
        """Create KnowledgeChunk from dictionary."""
        source = KnowledgeSource(data.get("source", "custom"))
        return cls(
            content=data.get("content", ""),
            source=source,
            title=data.get("title", ""),
            relevance_score=data.get("relevance_score", 0.0),
            url=data.get("url", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class KnowledgeQueryResult:
    """
    Result of a knowledge query containing multiple chunks.

    Aggregates multiple relevant chunks for a single query.
    """

    query: str  # The original query
    chunks: list[KnowledgeChunk] = field(default_factory=list)
    total_results: int = 0  # Total results before filtering/limiting
    query_time_ms: float = 0.0  # Time taken for the query

    def __str__(self) -> str:
        return f"Query: '{self.query}' - {len(self.chunks)} results"

    def is_empty(self) -> bool:
        """Check if the result has no chunks."""
        return len(self.chunks) == 0

    def get_top_chunks(self, n: int = 3) -> list[KnowledgeChunk]:
        """Get the top N most relevant chunks."""
        sorted_chunks = sorted(self.chunks, key=lambda c: c.relevance_score, reverse=True)
        return sorted_chunks[:n]

    def to_prompt_string(self, max_chunks: int = 3, max_tokens: int = 1000) -> str:
        """
        Format chunks as a string suitable for prompt augmentation.

        Args:
            max_chunks: Maximum number of chunks to include
            max_tokens: Approximate max token count (rough char/4 estimate)

        Returns:
            Formatted string with relevant knowledge
        """
        if self.is_empty():
            return ""

        lines = ["=== 관련 지식 ==="]
        char_count = 0
        max_chars = max_tokens * 4  # Rough estimate

        for chunk in self.get_top_chunks(max_chunks):
            chunk_str = chunk.to_prompt_string()
            if char_count + len(chunk_str) > max_chars:
                break
            lines.append(chunk_str)
            lines.append("")  # Empty line between chunks
            char_count += len(chunk_str)

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "chunks": [c.to_dict() for c in self.chunks],
            "total_results": self.total_results,
            "query_time_ms": self.query_time_ms,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeQueryResult":
        """Create KnowledgeQueryResult from dictionary."""
        chunks = [KnowledgeChunk.from_dict(c) for c in data.get("chunks", [])]
        return cls(
            query=data.get("query", ""),
            chunks=chunks,
            total_results=data.get("total_results", len(chunks)),
            query_time_ms=data.get("query_time_ms", 0.0),
        )

    @classmethod
    def empty(cls, query: str) -> "KnowledgeQueryResult":
        """Create an empty result for a query."""
        return cls(query=query, chunks=[], total_results=0)
