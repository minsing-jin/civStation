"""
Web Search Retriever - Knowledge retrieval from web search.

Retrieves knowledge by searching the web for Civilization VI related
information. Supports multiple search providers (Tavily, SerpAPI, etc.).
"""

import logging
import os
import time
from typing import Any

from computer_use_test.agent.modules.knowledge.base_retriever import BaseKnowledgeRetriever
from computer_use_test.agent.modules.knowledge.schemas.knowledge_schemas import (
    KnowledgeChunk,
    KnowledgeSource,
)

logger = logging.getLogger(__name__)


class WebSearchRetriever(BaseKnowledgeRetriever):
    """
    Retriever that searches the web for relevant information.

    Supports multiple search providers:
    - Tavily (default, optimized for RAG)
    - SerpAPI (Google search)
    - Custom providers

    Automatically prepends "Civilization 6" to queries for relevance.
    """

    def __init__(
        self,
        search_provider: str = "tavily",
        api_key: str | None = None,
        search_prefix: str = "Civilization 6",
        max_retries: int = 3,
        timeout: float = 10.0,
    ):
        """
        Initialize the web search retriever.

        Args:
            search_provider: Search provider to use ("tavily", "serpapi")
            api_key: API key for the search provider (or from env)
            search_prefix: Prefix to add to all queries
            max_retries: Maximum retry attempts on failure
            timeout: Request timeout in seconds
        """
        self.search_provider = search_provider.lower()
        self.search_prefix = search_prefix
        self.max_retries = max_retries
        self.timeout = timeout

        # Get API key from parameter or environment
        self.api_key = api_key or self._get_api_key_from_env()

        # Initialize provider-specific client
        self._client: Any = None
        self._initialize_client()

    def _get_api_key_from_env(self) -> str | None:
        """Get API key from environment variables."""
        env_vars = {
            "tavily": "TAVILY_API_KEY",
            "serpapi": "SERPAPI_API_KEY",
        }
        env_var = env_vars.get(self.search_provider)
        if env_var:
            return os.getenv(env_var)
        return None

    def _initialize_client(self) -> None:
        """Initialize the search provider client."""
        if not self.api_key:
            logger.warning(f"No API key for {self.search_provider}, web search disabled")
            return

        try:
            if self.search_provider == "tavily":
                self._initialize_tavily()
            elif self.search_provider == "serpapi":
                self._initialize_serpapi()
            else:
                logger.warning(f"Unknown search provider: {self.search_provider}")
        except ImportError as e:
            logger.warning(f"Failed to import {self.search_provider} client: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize {self.search_provider}: {e}")

    def _initialize_tavily(self) -> None:
        """Initialize Tavily client."""
        try:
            from tavily import TavilyClient

            self._client = TavilyClient(api_key=self.api_key)
            logger.info("Tavily client initialized")
        except ImportError:
            logger.warning("tavily-python not installed. Install with: pip install tavily-python")

    def _initialize_serpapi(self) -> None:
        """Initialize SerpAPI client."""
        try:
            from serpapi import GoogleSearch

            self._client = GoogleSearch
            logger.info("SerpAPI client initialized")
        except ImportError:
            logger.warning("google-search-results not installed. Install with: pip install google-search-results")

    def retrieve(self, query: str, top_k: int = 5) -> list[KnowledgeChunk]:
        """
        Search the web and retrieve relevant knowledge chunks.

        Args:
            query: Search query
            top_k: Maximum results to return

        Returns:
            List of KnowledgeChunk objects
        """
        if not self.is_available():
            logger.warning("Web search not available")
            return []

        # Add prefix for better Civ6-related results
        full_query = f"{self.search_prefix} {query}" if self.search_prefix else query

        for attempt in range(self.max_retries):
            try:
                if self.search_provider == "tavily":
                    return self._search_tavily(full_query, top_k)
                elif self.search_provider == "serpapi":
                    return self._search_serpapi(full_query, top_k)
                else:
                    return []
            except Exception as e:
                logger.warning(f"Search attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))  # Exponential backoff

        return []

    def _search_tavily(self, query: str, top_k: int) -> list[KnowledgeChunk]:
        """Search using Tavily API."""
        response = self._client.search(
            query=query,
            search_depth="basic",
            max_results=top_k,
            include_answer=False,
        )

        chunks = []
        results = response.get("results", [])

        for i, result in enumerate(results[:top_k]):
            # Calculate relevance score (Tavily provides score)
            score = result.get("score", 1.0 - (i * 0.1))

            chunk = KnowledgeChunk(
                content=result.get("content", ""),
                source=KnowledgeSource.WEB_SEARCH,
                title=result.get("title", "Web Result"),
                relevance_score=min(1.0, max(0.0, score)),
                url=result.get("url", ""),
                metadata={
                    "provider": "tavily",
                    "published_date": result.get("published_date", ""),
                },
            )
            chunks.append(chunk)

        return chunks

    def _search_serpapi(self, query: str, top_k: int) -> list[KnowledgeChunk]:
        """Search using SerpAPI (Google Search)."""
        params = {
            "q": query,
            "api_key": self.api_key,
            "num": top_k,
        }

        search = self._client(params)
        results = search.get_dict()

        chunks = []
        organic_results = results.get("organic_results", [])

        for i, result in enumerate(organic_results[:top_k]):
            # Calculate relevance score based on position
            score = 1.0 - (i * 0.15)

            chunk = KnowledgeChunk(
                content=result.get("snippet", ""),
                source=KnowledgeSource.WEB_SEARCH,
                title=result.get("title", "Web Result"),
                relevance_score=max(0.1, score),
                url=result.get("link", ""),
                metadata={
                    "provider": "serpapi",
                    "position": i + 1,
                },
            )
            chunks.append(chunk)

        return chunks

    def is_available(self) -> bool:
        """Check if web search is available."""
        return self._client is not None and self.api_key is not None

    def get_provider(self) -> str:
        """Get the current search provider name."""
        return self.search_provider


class MockWebSearchRetriever(BaseKnowledgeRetriever):
    """Mock web search retriever for testing."""

    def __init__(self):
        self._mock_results: list[KnowledgeChunk] = []

    def set_mock_results(self, results: list[KnowledgeChunk]) -> None:
        """Set mock results to return."""
        self._mock_results = results

    def retrieve(self, query: str, top_k: int = 5) -> list[KnowledgeChunk]:
        """Return mock results."""
        return self._mock_results[:top_k]

    def is_available(self) -> bool:
        """Mock is always available."""
        return True
