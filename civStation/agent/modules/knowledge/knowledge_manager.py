"""
Knowledge Manager - Unified interface for knowledge retrieval.

Provides a single entry point for querying multiple knowledge sources
(documents, web search) and augmenting prompts with relevant knowledge.
"""

import logging
import time
from typing import TYPE_CHECKING

from civStation.agent.modules.knowledge.base_retriever import BaseKnowledgeRetriever
from civStation.agent.modules.knowledge.schemas.knowledge_schemas import (
    KnowledgeChunk,
    KnowledgeQueryResult,
)

if TYPE_CHECKING:
    from civStation.utils.llm_provider.base import BaseVLMProvider

logger = logging.getLogger(__name__)


class KnowledgeManager:
    """
    Unified knowledge management for RAG-based assistance.

    Aggregates multiple retrievers (documents, web search) and provides:
    - Unified query interface
    - Prompt augmentation
    - Question answering with context
    """

    def __init__(
        self,
        document_retriever: BaseKnowledgeRetriever | None = None,
        web_retriever: BaseKnowledgeRetriever | None = None,
        vlm_provider: "BaseVLMProvider | None" = None,
    ):
        """
        Initialize the knowledge manager.

        Args:
            document_retriever: Retriever for local documents
            web_retriever: Retriever for web search
            vlm_provider: VLM provider for question answering
        """
        self.document_retriever = document_retriever
        self.web_retriever = web_retriever
        self.vlm_provider = vlm_provider

        # Track available sources
        self._available_sources: list[str] = []
        self._update_available_sources()

    def _update_available_sources(self) -> None:
        """Update list of available knowledge sources."""
        self._available_sources = []

        if self.document_retriever and self.document_retriever.is_available():
            self._available_sources.append("documents")

        if self.web_retriever and self.web_retriever.is_available():
            self._available_sources.append("web")

    def query(
        self,
        question: str,
        sources: list[str] | None = None,
        top_k: int = 5,
    ) -> KnowledgeQueryResult:
        """
        Query knowledge sources for relevant information.

        Args:
            question: The query/question
            sources: List of sources to query (None = all available)
                     Options: ["documents", "web"]
            top_k: Maximum results per source

        Returns:
            KnowledgeQueryResult with aggregated chunks
        """
        start_time = time.time()
        all_chunks: list[KnowledgeChunk] = []

        # Use all available sources if not specified
        if sources is None:
            sources = self._available_sources

        # Query document retriever
        if "documents" in sources and self.document_retriever:
            try:
                doc_chunks = self.document_retriever.retrieve(question, top_k)
                all_chunks.extend(doc_chunks)
                logger.debug(f"Retrieved {len(doc_chunks)} document chunks")
            except Exception as e:
                logger.error(f"Document retrieval failed: {e}")

        # Query web retriever
        if "web" in sources and self.web_retriever:
            try:
                web_chunks = self.web_retriever.retrieve(question, top_k)
                all_chunks.extend(web_chunks)
                logger.debug(f"Retrieved {len(web_chunks)} web chunks")
            except Exception as e:
                logger.error(f"Web retrieval failed: {e}")

        # Sort by relevance score
        all_chunks.sort(key=lambda c: c.relevance_score, reverse=True)

        # Deduplicate by content similarity (simple exact match)
        seen_contents: set[str] = set()
        unique_chunks: list[KnowledgeChunk] = []
        for chunk in all_chunks:
            # Use first 200 chars as dedup key
            content_key = chunk.content[:200].lower()
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                unique_chunks.append(chunk)

        query_time = (time.time() - start_time) * 1000

        return KnowledgeQueryResult(
            query=question,
            chunks=unique_chunks[:top_k],
            total_results=len(all_chunks),
            query_time_ms=query_time,
        )

    def augment_prompt(
        self,
        base_prompt: str,
        question: str,
        max_chunks: int = 3,
        max_tokens: int = 500,
    ) -> str:
        """
        Augment a prompt with relevant knowledge.

        Retrieves knowledge related to the question and appends it
        to the base prompt for RAG-style completion.

        Args:
            base_prompt: The original prompt
            question: Question to find knowledge for
            max_chunks: Maximum knowledge chunks to include
            max_tokens: Approximate max token count for knowledge section

        Returns:
            Augmented prompt with knowledge context
        """
        # Query for relevant knowledge
        result = self.query(question, top_k=max_chunks * 2)

        if result.is_empty():
            return base_prompt

        # Format knowledge for prompt
        knowledge_section = result.to_prompt_string(max_chunks=max_chunks, max_tokens=max_tokens)

        if not knowledge_section:
            return base_prompt

        # Append to prompt
        return f"{base_prompt}\n\n{knowledge_section}"

    def answer_question(
        self,
        question: str,
        context: str = "",
        sources: list[str] | None = None,
    ) -> str:
        """
        Answer a question using RAG.

        Retrieves relevant knowledge and uses VLM to generate an answer.

        Args:
            question: The question to answer
            context: Additional context (e.g., current game state)
            sources: Knowledge sources to use

        Returns:
            Generated answer string
        """
        if not self.vlm_provider:
            logger.warning("No VLM provider available for question answering")
            return "VLM provider not configured."

        # Retrieve relevant knowledge
        result = self.query(question, sources=sources)

        # Build prompt
        knowledge_section = result.to_prompt_string(max_chunks=5, max_tokens=1000)

        prompt = f"""질문에 대해 제공된 지식을 바탕으로 답변해주세요.

{"=== 현재 상황 ===" + chr(10) + context + chr(10) if context else ""}
{knowledge_section if knowledge_section else "관련 지식 없음"}

=== 질문 ===
{question}

=== 답변 ===
"""

        # Get answer from VLM
        try:
            content_parts = [self.vlm_provider._build_text_content(prompt)]
            response = self.vlm_provider._send_to_api(
                content_parts,
                temperature=0.3,
                max_tokens=1024,
            )
            return response.content
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return f"답변 생성 실패: {e}"

    def get_available_sources(self) -> list[str]:
        """Get list of available knowledge sources."""
        self._update_available_sources()
        return self._available_sources

    def is_available(self) -> bool:
        """Check if any knowledge source is available."""
        self._update_available_sources()
        return len(self._available_sources) > 0

    def add_retriever(
        self,
        retriever: BaseKnowledgeRetriever,
        source_type: str,
    ) -> None:
        """
        Add a retriever to the knowledge manager.

        Args:
            retriever: The retriever to add
            source_type: "documents" or "web"
        """
        if source_type == "documents":
            self.document_retriever = retriever
        elif source_type == "web":
            self.web_retriever = retriever
        else:
            logger.warning(f"Unknown source type: {source_type}")

        self._update_available_sources()

    def get_stats(self) -> dict:
        """Get statistics about knowledge sources."""
        stats = {
            "available_sources": self._available_sources,
            "has_vlm": self.vlm_provider is not None,
        }

        if self.document_retriever and hasattr(self.document_retriever, "get_document_count"):
            stats["document_count"] = self.document_retriever.get_document_count()

        if self.web_retriever and hasattr(self.web_retriever, "get_provider"):
            stats["web_provider"] = self.web_retriever.get_provider()

        return stats

    def __repr__(self) -> str:
        sources = ", ".join(self._available_sources) or "none"
        return f"KnowledgeManager(sources=[{sources}])"
