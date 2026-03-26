"""
Base Retriever - Abstract interface for knowledge retrieval.

Defines the abstract base class that all knowledge retrievers must implement.
"""

from abc import ABC, abstractmethod

from civStation.agent.modules.knowledge.schemas.knowledge_schemas import KnowledgeChunk


class BaseKnowledgeRetriever(ABC):
    """
    Abstract base class for knowledge retrievers.

    Knowledge retrievers fetch relevant information from various sources
    (documents, web, databases) based on a query.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[KnowledgeChunk]:
        """
        Retrieve knowledge chunks relevant to the query.

        Args:
            query: The search query
            top_k: Maximum number of results to return

        Returns:
            List of KnowledgeChunk objects sorted by relevance
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the retriever is available and ready to use.

        Returns:
            True if the retriever can be used, False otherwise
        """
        pass

    def get_name(self) -> str:
        """Get the name of this retriever."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        status = "available" if self.is_available() else "unavailable"
        return f"{self.get_name()}({status})"
