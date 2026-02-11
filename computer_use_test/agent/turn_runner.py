"""
Civilization VI Agent — CLI Entry Point.

Parses command-line arguments, initializes all components (providers,
context, strategy planner, knowledge manager, chat app), then delegates
to the execution functions in turn_executor.py.

Examples:
    # Basic usage
    python -m computer_use_test.agent.turn_runner --provider gemini

    # With HITL strategy refinement
    python -m computer_use_test.agent.turn_runner --provider gemini \
        --strategy "과학 승리에 집중하고 군사력 유지" --hitl

    # Autonomous mode with web search
    python -m computer_use_test.agent.turn_runner --provider gemini \
        --autonomous --enable-web-search

    # Different providers for router vs planner
    python -m computer_use_test.agent.turn_runner \
        --router-provider gemini --planner-provider claude
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from computer_use_test.agent.modules.context import ContextManager
from computer_use_test.agent.turn_executor import run_multi_turn, run_one_turn
from computer_use_test.utils.llm_provider import create_provider, get_available_providers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    available = get_available_providers()
    provider_choices = list(available.keys())

    parser = argparse.ArgumentParser(
        description="Run Civilization VI AI Agent for one or more turns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Same provider for router and planner
  python -m computer_use_test.agent.turn_runner --provider gemini

  # With HITL strategy refinement
  python -m computer_use_test.agent.turn_runner --provider gemini \\
      --strategy "과학 승리에 집중하고 군사력 유지" --hitl

  # Autonomous mode (no HITL, strategy generated from context)
  python -m computer_use_test.agent.turn_runner --provider gemini --autonomous

  # With knowledge retrieval
  python -m computer_use_test.agent.turn_runner --provider gemini \\
      --knowledge-index ./data/civopedia.json --enable-web-search

  # Different providers for router vs planner
  python -m computer_use_test.agent.turn_runner \\
      --router-provider gemini --router-model gemini-3.0-flash-preview \\
      --planner-provider claude --planner-model claude-sonnet-4-20250514

Available providers: {", ".join(provider_choices)}
        """,
    )

    # Shared default provider/model
    parser.add_argument(
        "--provider",
        "-p",
        default=None,
        choices=provider_choices,
        help="Default VLM provider for both router and planner",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Default model for both router and planner",
    )

    # Router-specific overrides
    parser.add_argument(
        "--router-provider",
        default=None,
        choices=provider_choices,
        help="VLM provider for routing (overrides --provider)",
    )
    parser.add_argument(
        "--router-model",
        default=None,
        help="Model for routing (overrides --model)",
    )

    # Planner-specific overrides
    parser.add_argument(
        "--planner-provider",
        default=None,
        choices=provider_choices,
        help="VLM provider for planning (overrides --provider)",
    )
    parser.add_argument(
        "--planner-model",
        default=None,
        help="Model for planning (overrides --model)",
    )

    # Execution parameters
    parser.add_argument(
        "--turns",
        "-t",
        type=int,
        default=1,
        help="Number of turns to execute (default: 1)",
    )
    parser.add_argument(
        "--range",
        type=int,
        default=1000,
        help="Coordinate normalization range (default: 1000)",
    )
    parser.add_argument(
        "--delay-action",
        type=float,
        default=0.5,
        help="Delay before executing each action in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--delay-turn",
        type=float,
        default=1.0,
        help="Delay between turns in seconds (default: 1.0)",
    )

    # Strategy parameters
    parser.add_argument(
        "--strategy",
        "-s",
        default=None,
        help="High-level strategy to guide action selection (e.g., '과학 승리에 집중')",
    )
    parser.add_argument(
        "--hitl",
        action="store_true",
        help="Enable HITL (Human-in-the-Loop) mode for strategy refinement",
    )
    parser.add_argument(
        "--autonomous",
        action="store_true",
        help="Enable fully autonomous mode (strategy generated from context, no HITL)",
    )
    parser.add_argument(
        "--hitl-mode",
        choices=["async"],
        default=None,
        help="HITL interrupt mode: async (Enter key pauses after current turn)",
    )
    parser.add_argument(
        "--input-mode",
        choices=["voice", "text", "auto", "chatapp"],
        default="text",
        help="HITL input mode: voice (microphone), text (terminal), auto, or chatapp (default: text)",
    )
    parser.add_argument(
        "--stt-provider",
        choices=["whisper", "google", "openai"],
        default="whisper",
        help="Speech-to-text provider for voice input (default: whisper)",
    )

    # Chat app parameters
    parser.add_argument(
        "--chatapp",
        choices=["discord"],
        default=None,
        help="Chat app platform for strategy discussion (e.g., discord)",
    )
    parser.add_argument(
        "--discord-token",
        type=str,
        default=None,
        help="Discord bot token (or set DISCORD_BOT_TOKEN env var)",
    )
    parser.add_argument(
        "--discord-channel",
        type=str,
        default=None,
        help="Discord channel ID to listen on",
    )
    parser.add_argument(
        "--discord-user",
        type=str,
        default=None,
        help="Discord user ID to accept input from",
    )
    parser.add_argument(
        "--enable-discussion",
        action="store_true",
        help="Enable multi-turn strategy discussion via chat app",
    )

    # Knowledge retrieval parameters
    parser.add_argument(
        "--knowledge-index",
        type=str,
        default=None,
        help="Path to Civopedia/document index JSON file",
    )
    parser.add_argument(
        "--enable-web-search",
        action="store_true",
        help="Enable web search for knowledge retrieval (requires TAVILY_API_KEY)",
    )

    args = parser.parse_args()

    # ==================== Resolve providers ====================

    router_provider_name = args.router_provider or args.provider
    router_model = args.router_model or args.model
    planner_provider_name = args.planner_provider or args.provider
    planner_model = args.planner_model or args.model

    if not router_provider_name:
        parser.error("--provider is required, or specify both --router-provider and --planner-provider")
    if not planner_provider_name:
        parser.error("--provider is required, or specify both --router-provider and --planner-provider")

    router_provider = create_provider(
        provider_name=router_provider_name,
        model=router_model,
    )
    logger.info(f"Router:  {router_provider.get_provider_name()} ({router_provider.model})")

    if planner_provider_name == router_provider_name and planner_model == router_model:
        planner_provider = router_provider
        logger.info("Planner: same as router (shared instance)")
    else:
        planner_provider = create_provider(
            provider_name=planner_provider_name,
            model=planner_model,
        )
        logger.info(f"Planner: {planner_provider.get_provider_name()} ({planner_provider.model})")

    # ==================== Context manager ====================

    ctx = ContextManager.get_instance()
    logger.info("ContextManager initialized")

    # ==================== Chat app ====================

    chatapp_provider = None
    discussion_engine = None
    chat_app = None

    if args.chatapp == "discord":
        import os

        from computer_use_test.utils.chatapp import create_chat_app

        discord_token = args.discord_token or os.environ.get("DISCORD_BOT_TOKEN", "")
        if not discord_token:
            parser.error("Discord bot token required: use --discord-token or set DISCORD_BOT_TOKEN env var")

        allowed_users = [args.discord_user] if args.discord_user else []
        allowed_channels = [args.discord_channel] if args.discord_channel else []

        chat_app = create_chat_app(
            "discord",
            bot_token=discord_token,
            allowed_user_ids=allowed_users,
            allowed_channel_ids=allowed_channels,
        )
        chat_app.start()
        logger.info(f"Discord chat app started (channel={args.discord_channel}, user={args.discord_user})")

        if args.input_mode == "text" and args.chatapp:
            args.input_mode = "chatapp"

        if args.discord_user:
            from computer_use_test.agent.modules.hitl import ChatAppInputProvider

            chatapp_provider = ChatAppInputProvider(
                chat_app=chat_app,
                user_id=args.discord_user,
                channel_id=args.discord_channel,
            )
            logger.info("ChatAppInputProvider initialized")

        if args.enable_discussion:
            from computer_use_test.agent.modules.discussion import StrategyDiscussion
            from computer_use_test.agent.modules.discussion.discord_handler import DiscordDiscussionHandler

            discussion_engine = StrategyDiscussion(
                vlm_provider=planner_provider,
                context_manager=ctx,
            )
            DiscordDiscussionHandler(
                chat_app=chat_app,
                discussion_engine=discussion_engine,
            )
            logger.info("Discussion engine and Discord handler initialized")

    # ==================== Strategy planner ====================

    strategy_planner = None
    if args.hitl or args.autonomous:
        from computer_use_test.agent.modules.strategy import StrategyPlanner

        hitl_mode = args.hitl and not args.autonomous
        strategy_planner = StrategyPlanner(
            vlm_provider=planner_provider,
            hitl_mode=hitl_mode,
            input_mode=args.input_mode,
            stt_provider=args.stt_provider,
            chatapp_provider=chatapp_provider,
            discussion_engine=discussion_engine,
        )
        mode_str = "HITL" if hitl_mode else "autonomous"
        input_mode_str = f", input={args.input_mode}" if hitl_mode else ""
        logger.info(f"StrategyPlanner initialized ({mode_str} mode{input_mode_str})")

    # ==================== Knowledge manager ====================

    knowledge_manager = None
    if args.knowledge_index or args.enable_web_search:
        from computer_use_test.agent.modules.knowledge import (
            DocumentRetriever,
            KnowledgeManager,
            WebSearchRetriever,
        )

        doc_retriever = None
        web_retriever = None

        if args.knowledge_index:
            index_path = Path(args.knowledge_index)
            doc_retriever = DocumentRetriever(index_path=index_path)
            logger.info(f"DocumentRetriever initialized from {index_path}")

        if args.enable_web_search:
            web_retriever = WebSearchRetriever(search_provider="tavily")
            if web_retriever.is_available():
                logger.info("WebSearchRetriever initialized (Tavily)")
            else:
                logger.warning("WebSearchRetriever not available (check TAVILY_API_KEY)")
                web_retriever = None

        knowledge_manager = KnowledgeManager(
            document_retriever=doc_retriever,
            web_retriever=web_retriever,
            vlm_provider=planner_provider,
        )
        logger.info(f"KnowledgeManager initialized: {knowledge_manager}")

    # ==================== Run ====================

    try:
        if args.turns == 1:
            run_one_turn(
                router_provider=router_provider,
                planner_provider=planner_provider,
                normalizing_range=args.range,
                delay_before_action=args.delay_action,
                high_level_strategy=args.strategy,
                context_manager=ctx,
                strategy_planner=strategy_planner,
                knowledge_manager=knowledge_manager,
            )
        else:
            run_multi_turn(
                router_provider=router_provider,
                planner_provider=planner_provider,
                num_turns=args.turns,
                normalizing_range=args.range,
                delay_between_turns=args.delay_turn,
                delay_before_action=args.delay_action,
                high_level_strategy=args.strategy,
                context_manager=ctx,
                strategy_planner=strategy_planner,
                knowledge_manager=knowledge_manager,
                hitl_mode=args.hitl_mode,
            )
    finally:
        if chat_app:
            try:
                chat_app.stop()
                logger.info("Chat app stopped")
            except Exception as e:
                logger.warning(f"Error stopping chat app: {e}")


if __name__ == "__main__":
    main()
