"""
Knowledge Module - RAG-based knowledge retrieval for Civilization VI agent.

Provides:
- KnowledgeManager: Unified interface for knowledge retrieval
- DocumentRetriever: Local document retrieval with optional embeddings
- WebSearchRetriever: Web search for real-time information
- Knowledge schemas for structured knowledge representation
"""

from computer_use_test.agent.modules.knowledge.base_retriever import BaseKnowledgeRetriever
from computer_use_test.agent.modules.knowledge.document_retriever import DocumentRetriever
from computer_use_test.agent.modules.knowledge.knowledge_manager import KnowledgeManager
from computer_use_test.agent.modules.knowledge.schemas.knowledge_schemas import (
    KnowledgeChunk,
    KnowledgeQueryResult,
    KnowledgeSource,
)
from computer_use_test.agent.modules.knowledge.web_search_retriever import (
    MockWebSearchRetriever,
    WebSearchRetriever,
)

__all__ = [
    # Manager
    "KnowledgeManager",
    # Retrievers
    "BaseKnowledgeRetriever",
    "DocumentRetriever",
    "WebSearchRetriever",
    "MockWebSearchRetriever",
    # Schemas
    "KnowledgeSource",
    "KnowledgeChunk",
    "KnowledgeQueryResult",
]
