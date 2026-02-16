"""
Civilization VI Agent — CLI Entry Point with ConfigArgParse.
Loads default settings from config.yaml, overridable via CLI.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import configargparse  # pip install configargparse

from computer_use_test.agent.modules.context import ContextManager
from computer_use_test.agent.turn_executor import run_multi_turn, run_one_turn
from computer_use_test.utils.llm_provider import create_provider, get_available_providers

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> configargparse.Namespace:
    """ConfigArgParse를 사용하여 YAML과 CLI 인자를 동시에 처리"""
    available = get_available_providers()
    provider_choices = list(available.keys())

    parser = configargparse.ArgumentParser(
        description="Run Civilization VI AI Agent",
        # 기본적으로 읽을 설정 파일 지정
        default_config_files=["config.yaml"],
        # YAML 파서 명시 (YAML 문법 지원)
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.RawDescriptionHelpFormatter,
        epilog="Use python -m computer_use_test.agent.turn_runner --help to see full options.",
    )

    # 설정 파일 경로를 직접 지정할 수 있는 옵션 (--config my_config.yaml)
    parser.add_argument("--config", is_config_file=True, help="Path to configuration file")

    # --- Group 1: Provider Settings ---
    provider_group = parser.add_argument_group("LLM Provider Configuration")
    provider_group.add_argument("--provider", "-p", choices=provider_choices, help="Default provider")
    provider_group.add_argument("--model", "-m", help="Default model name")
    provider_group.add_argument("--router-provider", choices=provider_choices, help="Override provider for routing")
    provider_group.add_argument("--router-model", help="Override model for routing")
    provider_group.add_argument("--planner-provider", choices=provider_choices, help="Override provider for planning")
    provider_group.add_argument("--planner-model", help="Override model for planning")

    # --- Group 2: Execution Settings ---
    exec_group = parser.add_argument_group("Execution Parameters")
    exec_group.add_argument("--turns", "-t", type=int, default=1, help="Number of turns")
    exec_group.add_argument("--range", type=int, default=1000, help="Coordinate normalization range")
    exec_group.add_argument("--delay-action", type=float, default=0.5, help="Action delay (sec)")
    exec_group.add_argument("--delay-turn", type=float, default=1.0, help="Turn delay (sec)")

    # --- Group 3: Strategy & HITL ---
    strat_group = parser.add_argument_group("Strategy & Human-in-the-Loop")
    strat_group.add_argument("--strategy", "-s", help="High-level strategy guidance")
    strat_group.add_argument("--hitl", action="store_true", help="Enable HITL mode")
    strat_group.add_argument("--autonomous", action="store_true", help="Enable autonomous mode")
    strat_group.add_argument("--hitl-mode", choices=["async"], help="Interrupt mode")
    strat_group.add_argument("--input-mode", choices=["voice", "text", "auto", "chatapp"], default="text")
    strat_group.add_argument("--stt-provider", choices=["whisper", "google", "openai"], default="whisper")

    # --- Group 4: Chat App (Discord) ---
    chat_group = parser.add_argument_group("Chat App Integration")
    chat_group.add_argument("--chatapp", choices=["discord"], help="Chat platform")
    chat_group.add_argument("--discord-token", help="Discord bot token")
    chat_group.add_argument("--discord-channel", help="Discord channel ID")
    chat_group.add_argument("--discord-user", help="Discord user ID")
    chat_group.add_argument("--enable-discussion", action="store_true", help="Enable strategy discussion")

    # --- Group 5: Knowledge ---
    know_group = parser.add_argument_group("Knowledge Retrieval")
    know_group.add_argument("--knowledge-index", help="Path to Civopedia JSON")
    know_group.add_argument("--enable-web-search", action="store_true", help="Enable Tavily web search")

    return parser.parse_args()


def setup_providers(args) -> tuple[object, object]:
    """Router와 Planner Provider 초기화"""
    router_p_name = args.router_provider or args.provider
    router_model = args.router_model or args.model
    planner_p_name = args.planner_provider or args.provider
    planner_model = args.planner_model or args.model

    if not router_p_name or not planner_p_name:
        raise ValueError("Provider not specified. Check config.yaml or use --provider CLI arg.")

    # Router 생성
    router = create_provider(provider_name=router_p_name, model=router_model)
    logger.info(f"Router initialized: {router.get_provider_name()} ({router.model})")

    # Planner 생성
    if planner_p_name == router_p_name and planner_model == router_model:
        planner = router
        logger.info("Planner: Sharing instance with Router")
    else:
        planner = create_provider(provider_name=planner_p_name, model=planner_model)
        logger.info(f"Planner initialized: {planner.get_provider_name()} ({planner.model})")

    return router, planner


def setup_chat_app(args, planner_provider, context_manager):
    """Discord 앱 초기화"""
    if args.chatapp != "discord":
        return None, None, None

    from computer_use_test.agent.modules.hitl import ChatAppInputProvider
    from computer_use_test.utils.chatapp import create_chat_app

    token = args.discord_token or os.environ.get("DISCORD_BOT_TOKEN", "")
    if not token:
        raise ValueError("Discord token missing. Set DISCORD_BOT_TOKEN or check config.yaml")

    # 1. Start App
    chat_app = create_chat_app(
        "discord",
        bot_token=token,
        allowed_user_ids=[args.discord_user] if args.discord_user else [],
        allowed_channel_ids=[args.discord_channel] if args.discord_channel else [],
    )
    chat_app.start()
    logger.info(f"Discord app started (Channel: {args.discord_channel})")

    # 2. Input Provider
    chatapp_provider = None
    if args.discord_user:
        if args.input_mode == "text":
            args.input_mode = "chatapp"  # Discord 사용자가 지정되면 chatapp 모드 우선
        chatapp_provider = ChatAppInputProvider(chat_app=chat_app, user_id=args.discord_user, channel_id=args.discord_channel)

    # 3. Discussion Engine
    discussion_engine = None
    if args.enable_discussion:
        from computer_use_test.utils.chatapp.discussion import DiscordDiscussionHandler, StrategyDiscussion

        discussion_engine = StrategyDiscussion(vlm_provider=planner_provider, context_manager=context_manager)
        DiscordDiscussionHandler(chat_app=chat_app, discussion_engine=discussion_engine)

    return chat_app, chatapp_provider, discussion_engine


def setup_knowledge(args, vlm_provider):
    """지식 검색 모듈 초기화"""
    if not (args.knowledge_index or args.enable_web_search):
        return None

    from computer_use_test.agent.modules.knowledge import DocumentRetriever, KnowledgeManager, WebSearchRetriever

    doc_retriever = DocumentRetriever(Path(args.knowledge_index)) if args.knowledge_index else None

    web_retriever = None
    if args.enable_web_search:
        web_retriever = WebSearchRetriever(search_provider="tavily")
        if not web_retriever.is_available():
            logger.warning("Web search disabled: API key missing")
            web_retriever = None

    km = KnowledgeManager(
        document_retriever=doc_retriever,
        web_retriever=web_retriever,
        vlm_provider=vlm_provider,
    )
    logger.info("KnowledgeManager initialized")
    return km


def main():
    # 1. Parse Arguments (YAML + CLI)
    try:
        args = parse_args()
    except SystemExit:
        return

    # 2. Setup Components
    try:
        router_provider, planner_provider = setup_providers(args)
        ctx = ContextManager.get_instance()

        chat_app, chatapp_input_provider, discussion_engine = setup_chat_app(args, planner_provider, ctx)

        strategy_planner = None
        if args.hitl or args.autonomous:
            from computer_use_test.agent.modules.strategy import StrategyPlanner

            strategy_planner = StrategyPlanner(
                vlm_provider=planner_provider,
                hitl_mode=(args.hitl and not args.autonomous),
                input_mode=args.input_mode,
                stt_provider=args.stt_provider,
                chatapp_provider=chatapp_input_provider,
                discussion_engine=discussion_engine,
            )

        knowledge_manager = setup_knowledge(args, planner_provider)

    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        if "chat_app" in locals() and chat_app:
            chat_app.stop()
        return

    # 3. Execution
    logger.info(f"Starting execution for {args.turns} turn(s)...")
    try:
        runner_kwargs = {
            "router_provider": router_provider,
            "planner_provider": planner_provider,
            "normalizing_range": args.range,
            "delay_before_action": args.delay_action,
            "high_level_strategy": args.strategy,
            "context_manager": ctx,
            "strategy_planner": strategy_planner,
            "knowledge_manager": knowledge_manager,
        }

        if args.turns == 1:
            run_one_turn(**runner_kwargs)
        else:
            run_multi_turn(num_turns=args.turns, delay_between_turns=args.delay_turn, hitl_mode=args.hitl_mode, **runner_kwargs)
    finally:
        if chat_app:
            chat_app.stop()
            logger.info("Chat app stopped")


if __name__ == "__main__":
    main()
