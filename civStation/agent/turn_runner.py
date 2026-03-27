"""
Civilization VI Agent — CLI Entry Point with ConfigArgParse.
Loads default settings from config.yaml, overridable via CLI.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import configargparse  # pip install configargparse

from civStation.agent.modules.context import ContextManager
from civStation.agent.modules.hitl import CommandQueue, QueueListener
from civStation.agent.turn_executor import run_multi_turn, run_one_turn
from civStation.utils.debug import DebugOptions
from civStation.utils.llm_provider import create_provider, get_available_providers
from civStation.utils.run_log_cache import start_run_log_session
from civStation.utils.screenshot_trajectory import start_screenshot_trajectory_session

# Logging setup (로깅 설정)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Suppress noisy third-party loggers
for _noisy in ("httpx", "httpcore", "urllib3", "asyncio", "websockets"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class _ConsoleLogSilencer:
    """Temporarily suppress INFO/DEBUG logs on console handlers only."""

    def __init__(self, logger: logging.Logger | None = None, minimum_level: int = logging.WARNING) -> None:
        self._logger = logger or logging.getLogger()
        self._minimum_level = minimum_level
        self._original_levels: list[tuple[logging.Handler, int]] = []

    def enable(self) -> None:
        if self._original_levels:
            return

        for handler in self._logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                self._original_levels.append((handler, handler.level))
                if handler.level == logging.NOTSET or handler.level < self._minimum_level:
                    handler.setLevel(self._minimum_level)

    def disable(self) -> None:
        while self._original_levels:
            handler, level = self._original_levels.pop()
            handler.setLevel(level)


def parse_args(argv: list[str] | None = None) -> configargparse.Namespace:
    """ConfigArgParse를 사용하여 YAML과 CLI 인자를 동시에 처리"""
    available = get_available_providers()
    provider_choices = list(available.keys())

    parser = configargparse.ArgumentParser(
        description="Run Civilization VI AI Agent",
        # Default config file (기본 설정 파일)
        default_config_files=["config.yaml"],
        # Use YAML parser (YAML 문법 지원)
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.RawDescriptionHelpFormatter,
        epilog=(
            "Recommended entrypoint: `uv run civstation run ...` or `civstation run ...`. "
            "Legacy module form still works: `python -m civStation.agent.turn_runner --help`."
        ),
    )

    # Override config file path (설정 파일 경로를 직접 지정)
    parser.add_argument("--config", is_config_file=True, help="Path to configuration file")

    # --- Group 1: Provider Settings ---
    provider_group = parser.add_argument_group("LLM Provider Configuration")
    provider_group.add_argument("--provider", "-p", choices=provider_choices, help="Default provider")
    provider_group.add_argument("--model", "-m", help="Default model name")
    provider_group.add_argument("--router-provider", choices=provider_choices, help="Override provider for routing")
    provider_group.add_argument("--router-model", help="Override model for routing")
    provider_group.add_argument("--planner-provider", choices=provider_choices, help="Override provider for planning")
    provider_group.add_argument("--planner-model", help="Override model for planning")
    provider_group.add_argument(
        "--turn-detector-provider", choices=provider_choices, help="Override provider for turn detection"
    )
    provider_group.add_argument("--turn-detector-model", help="Override model for turn detection")

    # --- Group 2: Execution Settings ---
    exec_group = parser.add_argument_group("Execution Parameters")
    exec_group.add_argument("--turns", "-t", type=int, default=1, help="Number of turns")
    exec_group.add_argument("--range", type=int, default=1000, help="Coordinate normalization range")
    exec_group.add_argument("--delay-action", type=float, default=0.5, help="Action delay (sec)")
    exec_group.add_argument("--delay-turn", type=float, default=1.0, help="Turn delay (sec)")
    exec_group.add_argument(
        "--prompt-language",
        choices=["eng", "kor"],
        default="eng",
        help="Primitive prompt language (default: eng)",
    )
    exec_group.add_argument(
        "--debug",
        default="",
        help=(
            "Comma-separated debug features to enable. Options: context (log full context string), "
            "turns (turn-number validation). Use 'all' to enable everything. Example: --debug context,turns"
        ),
    )

    # --- Group 3: Strategy & HITL ---
    strat_group = parser.add_argument_group("Strategy & Human-in-the-Loop")
    strat_group.add_argument("--strategy", "-s", help="High-level strategy guidance")
    strat_group.add_argument("--hitl", action="store_true", help="Enable HITL mode")
    strat_group.add_argument("--autonomous", action="store_true", help="Enable autonomous mode")
    strat_group.add_argument("--hitl-mode", choices=["async"], help="Interrupt mode")

    # --- Group 4: Chat App ---
    chat_group = parser.add_argument_group("Chat App Integration")
    chat_group.add_argument(
        "--chatapp",
        choices=["original", "discord", "whatsapp"],
        help="Chat platform for HITL input: 'original' (built-in status UI), 'discord', or 'whatsapp'",
    )
    chat_group.add_argument("--discord-token", help="Discord bot token")
    chat_group.add_argument("--discord-channel", help="Discord channel ID")
    chat_group.add_argument("--discord-user", help="Discord user ID")
    chat_group.add_argument("--whatsapp-token", help="WhatsApp bot token")
    chat_group.add_argument("--whatsapp-phone-number-id", help="WhatsApp phone number ID")
    chat_group.add_argument("--whatsapp-user", help="WhatsApp user ID to receive messages from")
    chat_group.add_argument("--enable-discussion", action="store_true", help="Enable strategy discussion")

    # --- Group 5: Knowledge ---
    know_group = parser.add_argument_group("Knowledge Retrieval")
    know_group.add_argument("--knowledge-index", help="Path to Civopedia JSON")
    know_group.add_argument("--enable-web-search", action="store_true", help="Enable Tavily web search")

    # --- Group 6: Control API & Status UI ---
    ui_group = parser.add_argument_group("Control API & Status UI")
    ui_group.add_argument(
        "--status-ui", action="store_true", help="Enable real-time status dashboard (includes control API)"
    )
    ui_group.add_argument("--control-api", action="store_true", help="Enable start/pause/resume/stop API server")
    ui_group.add_argument("--status-port", type=int, default=8765, help="Control API / Status UI port (default: 8765)")
    ui_group.add_argument(
        "--wait-for-start",
        action="store_true",
        help="Wait for external start signal via API before running "
        "(requires --control-api/--status-ui or --relay-url)",
    )

    # --- Group 7: Image Pipeline ---
    img_group = parser.add_argument_group("Image Pipeline (per-site preprocessing)")
    for _site in ("router", "planner", "context", "turn-detector"):
        img_group.add_argument(f"--{_site}-img-preset", help=f"Preset name for {_site} images")
        img_group.add_argument(f"--{_site}-img-max-long-edge", type=int, help=f"Max long edge for {_site}")
        img_group.add_argument(f"--{_site}-img-ui-filter", help=f"UI filter mode for {_site}")
        img_group.add_argument(f"--{_site}-img-color", help=f"Color policy for {_site}")
        img_group.add_argument(f"--{_site}-img-encode", help=f"Encode mode for {_site}")
        img_group.add_argument(f"--{_site}-img-jpeg-quality", type=int, help=f"JPEG quality for {_site}")

    # --- Group 8: Relay ---
    relay_group = parser.add_argument_group("Relay (remote HITL via external relay server)")
    relay_group.add_argument("--relay-url", help="WebSocket URL of the relay server (wss://...)")
    relay_group.add_argument(
        "--relay-token",
        help="Auth token for the relay server (or set RELAY_TOKEN env var)",
    )

    return parser.parse_args(argv)


def setup_providers(args) -> tuple[object, object]:
    """Router와 Planner Provider 초기화"""
    router_p_name = args.router_provider or args.provider
    router_model = args.router_model or args.model
    planner_p_name = args.planner_provider or args.provider
    planner_model = args.planner_model or args.model

    if not router_p_name or not planner_p_name:
        raise ValueError("Provider not specified. Check config.yaml or use --provider CLI arg.")

    # Create router provider (Router 생성)
    router = create_provider(provider_name=router_p_name, model=router_model)
    logger.info(f"Router initialized: {router.get_provider_name()} ({router.model})")

    # Create planner provider (Planner 생성)
    if planner_p_name == router_p_name and planner_model == router_model:
        planner = router
        logger.info("Planner: Sharing instance with Router")
    else:
        planner = create_provider(provider_name=planner_p_name, model=planner_model)
        logger.info(f"Planner initialized: {planner.get_provider_name()} ({planner.model})")

    return router, planner


def setup_chat_app(args, planner_provider, context_manager):
    """채팅 앱 초기화 (original / discord / whatsapp)"""
    platform = getattr(args, "chatapp", None)

    if not platform or platform == "original":
        # Built-in mode: no external bot, HITL handled via status UI / command queue
        return None, None, None

    from civStation.agent.modules.hitl import ChatAppInputProvider
    from civStation.utils.chatapp import create_chat_app

    if platform == "discord":
        token = args.discord_token or os.environ.get("DISCORD_BOT_TOKEN", "")
        if not token:
            raise ValueError("Discord token missing. Set DISCORD_BOT_TOKEN or --discord-token")

        chat_app = create_chat_app(
            "discord",
            bot_token=token,
            allowed_user_ids=[args.discord_user] if args.discord_user else [],
            allowed_channel_ids=[args.discord_channel] if args.discord_channel else [],
        )
        chat_app.start()
        logger.info(f"Discord app started (Channel: {args.discord_channel})")

        chatapp_provider = None
        if args.discord_user:
            chatapp_provider = ChatAppInputProvider(
                chat_app=chat_app,
                user_id=args.discord_user,
                channel_id=args.discord_channel,
            )

    elif platform == "whatsapp":
        token = args.whatsapp_token or os.environ.get("WHATSAPP_BOT_TOKEN", "")
        if not token:
            raise ValueError("WhatsApp token missing. Set WHATSAPP_BOT_TOKEN or --whatsapp-token")

        chat_app = create_chat_app(
            "whatsapp",
            bot_token=token,
            phone_number_id=getattr(args, "whatsapp_phone_number_id", "") or "",
            allowed_user_ids=[args.whatsapp_user] if getattr(args, "whatsapp_user", None) else [],
        )
        chat_app.start()
        logger.info("WhatsApp app started")

        chatapp_provider = None
        if getattr(args, "whatsapp_user", None):
            chatapp_provider = ChatAppInputProvider(
                chat_app=chat_app,
                user_id=args.whatsapp_user,
            )

    else:
        raise ValueError(f"Unknown chatapp platform: {platform}")

    # Discussion Engine (platform-agnostic)
    discussion_engine = None
    if args.enable_discussion:
        from civStation.utils.chatapp.discussion import DiscordDiscussionHandler, StrategyDiscussion

        discussion_engine = StrategyDiscussion(vlm_provider=planner_provider, context_manager=context_manager)
        DiscordDiscussionHandler(chat_app=chat_app, discussion_engine=discussion_engine)

    return chat_app, chatapp_provider, discussion_engine


def setup_knowledge(args, vlm_provider):
    """지식 검색 모듈 초기화"""
    if not (args.knowledge_index or args.enable_web_search):
        return None

    from civStation.agent.modules.knowledge import DocumentRetriever, KnowledgeManager, WebSearchRetriever

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


def main(argv: list[str] | None = None):
    console_log_silencer = _ConsoleLogSilencer()
    console_log_silencer.enable()
    try:
        return _main(argv, console_log_silencer)
    finally:
        console_log_silencer.disable()


def _main(argv: list[str] | None, console_log_silencer: _ConsoleLogSilencer):
    # 1. Parse Arguments (YAML + CLI)
    try:
        args = parse_args(argv) if argv is not None else parse_args()
    except SystemExit:
        return

    run_log_session = None
    try:
        run_log_session = start_run_log_session()
        logger.info("Raw run log cache initialized: %s", run_log_session.path)
    except Exception as e:
        logger.warning("Raw run log cache disabled: %s", e)

    trajectory_session = None
    try:
        trajectory_session = start_screenshot_trajectory_session(max_images=20)
        logger.info("Screenshot trajectory session initialized: %s", trajectory_session.path)
    except Exception as e:
        logger.warning("Screenshot trajectory disabled: %s", e)

    def close_run_log_session() -> None:
        nonlocal run_log_session
        if run_log_session is not None:
            run_log_session.close()
            run_log_session = None

    def close_trajectory_session() -> None:
        nonlocal trajectory_session
        if trajectory_session is not None:
            trajectory_session.close()
            trajectory_session = None

    # 2. Setup Components
    try:
        router_provider, planner_provider = setup_providers(args)
        ctx = ContextManager.get_instance()

        chat_app, chatapp_input_provider, discussion_engine = setup_chat_app(args, planner_provider, ctx)

        strategy_planner = None
        if args.hitl or args.autonomous:
            from civStation.agent.modules.strategy import StrategyPlanner

            strategy_planner = StrategyPlanner(
                vlm_provider=planner_provider,
                hitl_mode=(args.hitl and not args.autonomous),
                chatapp_provider=chatapp_input_provider,
                discussion_engine=discussion_engine,
            )

        knowledge_manager = setup_knowledge(args, planner_provider)

    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        if "chat_app" in locals() and chat_app:
            chat_app.stop()
        close_run_log_session()
        close_trajectory_session()
        return

    # 2b. Image Pipeline Configs
    from civStation.utils.image_pipeline import config_from_args as img_config_from_args

    router_img_config = img_config_from_args(args, "router")
    planner_img_config = img_config_from_args(args, "planner")
    context_img_config = img_config_from_args(args, "context")
    turn_detector_img_config = img_config_from_args(args, "turn_detector")

    # 3. Command Queue + Queue Listener (Phase 1)
    command_queue = CommandQueue()
    queue_listener = None
    if args.hitl_mode == "async":
        from civStation.agent.modules.hitl import HITLInputManager

        input_manager = HITLInputManager(chatapp_provider=chatapp_input_provider)
        queue_listener = QueueListener(command_queue, input_manager)
        queue_listener.start()

    # 4. Macro-Turn Manager (Phase 2)
    macro_turn_manager = None
    try:
        from civStation.agent.modules.context.macro_turn_manager import MacroTurnManager

        macro_turn_manager = MacroTurnManager(ctx, planner_provider)
        logger.info("MacroTurnManager initialized")
    except Exception as e:
        logger.warning(f"MacroTurnManager init failed: {e}")

    # 4b. Context Updater (background game-state analysis)
    context_updater = None
    try:
        from civStation.agent.modules.context.context_updater import ContextUpdater

        context_updater = ContextUpdater(ctx, router_provider, img_config=context_img_config)
        context_updater.start()
        logger.info("ContextUpdater background worker started")
    except Exception as e:
        logger.warning(f"ContextUpdater init failed: {e}")

    # 4d. Turn Detector (background turn-number detection)
    # Calibration is lazy — happens automatically on first submit()
    # when the game screen is actually visible.
    turn_detector = None
    td_provider_name = getattr(args, "turn_detector_provider", None) or args.router_provider or args.provider
    td_model = getattr(args, "turn_detector_model", None) or args.router_model or args.model
    if td_provider_name:
        try:
            from civStation.agent.modules.context.turn_detector import TurnDetector

            td_provider = create_provider(provider_name=td_provider_name, model=td_model)
            turn_detector = TurnDetector(td_provider, img_config=turn_detector_img_config)
            turn_detector.start()
            logger.info("TurnDetector background worker started (calibration deferred to first game screenshot)")
        except Exception as e:
            logger.warning(f"TurnDetector init failed: {e}")
            turn_detector = None

    # 4c. Strategy Updater (background strategy generation)
    strategy_updater = None
    if strategy_planner:
        try:
            from civStation.agent.modules.strategy.strategy_updater import (
                StrategyRequest,
                StrategyTrigger,
                StrategyUpdater,
            )

            strategy_updater = StrategyUpdater(ctx, strategy_planner)
            strategy_updater.start()
            # Submit initial strategy request
            strategy_updater.submit(StrategyRequest(StrategyTrigger.INITIAL, human_input=args.strategy))
            logger.info("StrategyUpdater background worker started")
        except Exception as e:
            logger.warning(f"StrategyUpdater init failed: {e}")

    # 5. Agent Gate + Status UI (Phase 3)
    agent_gate = None
    state_bridge = None
    status_server = None

    from civStation.agent.modules.hitl.agent_gate import AgentGate

    agent_gate = AgentGate(command_queue)

    relay_client = None
    enable_control_api = bool(args.status_ui or getattr(args, "control_api", False))

    if enable_control_api:
        try:
            from civStation.agent.modules.hitl.status_ui.server import StatusServer
            from civStation.agent.modules.hitl.status_ui.state_bridge import AgentStateBridge
            from civStation.agent.modules.hitl.status_ui.websocket_manager import WebSocketManager

            ws_manager = WebSocketManager()

            # Relay client (optional — only if --relay-url is provided)
            relay_url = getattr(args, "relay_url", None)
            if relay_url:
                from civStation.agent.modules.hitl.relay import RelayClient

                relay_token = getattr(args, "relay_token", None) or os.environ.get("RELAY_TOKEN", "")
                relay_client = RelayClient(
                    url=relay_url,
                    token=relay_token,
                    agent_gate=agent_gate,
                    command_queue=command_queue,
                )
                relay_client.start()
                logger.info(f"RelayClient started → {relay_url}")

            state_bridge = AgentStateBridge(
                ctx,
                command_queue,
                ws_manager=ws_manager,
                agent_gate=agent_gate,
                relay_client=relay_client,
            )
            # Create discussion engine for web-based strategy discussion
            web_discussion_engine = discussion_engine
            if not web_discussion_engine:
                from civStation.utils.chatapp.discussion import StrategyDiscussion

                web_discussion_engine = StrategyDiscussion(vlm_provider=planner_provider, context_manager=ctx)

            status_server = StatusServer(
                state_bridge,
                command_queue,
                ws_manager=ws_manager,
                agent_gate=agent_gate,
                discussion_engine=web_discussion_engine,
                port=args.status_port,
            )
            status_server.start()
            if args.status_ui:
                logger.info(f"Status UI available at http://localhost:{args.status_port}")
            else:
                logger.info(f"Control API available at http://localhost:{args.status_port}")
        except ImportError:
            logger.warning("Control API requires fastapi and uvicorn. Install with: pip install 'civStation[ui]'")
        except Exception as e:
            logger.warning(f"Control API init failed: {e}")
    elif getattr(args, "relay_url", None):
        # Relay without status UI (headless mode)
        try:
            from civStation.agent.modules.hitl.relay import RelayClient
            from civStation.agent.modules.hitl.status_ui.state_bridge import AgentStateBridge

            relay_token = getattr(args, "relay_token", None) or os.environ.get("RELAY_TOKEN", "")
            relay_client = RelayClient(
                url=args.relay_url,
                token=relay_token,
                agent_gate=agent_gate,
                command_queue=command_queue,
            )
            relay_client.start()
            state_bridge = AgentStateBridge(ctx, command_queue, agent_gate=agent_gate, relay_client=relay_client)
            logger.info(f"RelayClient (headless) started → {args.relay_url}")
        except Exception as e:
            logger.warning(f"Relay init failed: {e}")

    # 6. Execution
    debug_options = DebugOptions.from_str(getattr(args, "debug", ""))
    if debug_options.any_enabled():
        logger.info(f"Debug features enabled: {debug_options}")

    # Wait for external start signal.
    # In async HITL mode, default to waiting to prevent unintended autonomous execution.
    should_wait_for_start = bool(args.wait_for_start)
    if not should_wait_for_start and args.hitl_mode == "async":
        should_wait_for_start = True
        logger.info("Async HITL mode detected: auto-waiting for external start signal.")

    if should_wait_for_start:
        has_status_control = bool(enable_control_api or getattr(args, "relay_url", None))
        has_chatapp_control = bool(chatapp_input_provider is not None)

        if not (has_status_control or has_chatapp_control):
            logger.error(
                "Manual start required, but no control channel is enabled. "
                "Use --control-api/--status-ui, --relay-url, or chatapp input provider."
            )
            close_run_log_session()
            close_trajectory_session()
            return
        elif has_status_control:
            logger.info("Agent ready. Waiting for external start signal (POST /api/agent/start)...")
            if state_bridge:
                state_bridge.broadcast_agent_phase("대기 중 — 시작 신호 대기")
            agent_gate.wait_for_start()
            if agent_gate.is_stopped:
                logger.info("STOP received before start. Exiting.")
                close_run_log_session()
                close_trajectory_session()
                return
            console_log_silencer.disable()
            logger.info("Start signal received. Beginning execution.")
        else:
            from civStation.agent.modules.hitl.agent_gate import AgentState
            from civStation.agent.modules.hitl.command_queue import DirectiveType

            logger.info("Agent ready. Waiting for chatapp start command (resume/start)...")
            if state_bridge:
                state_bridge.broadcast_agent_phase("대기 중 — chatapp 시작 신호 대기")

            while True:
                command_queue.wait(timeout=None)
                pending = command_queue.drain()
                should_start = False
                should_stop = False
                keep: list = []

                for d in pending:
                    if d.directive_type == DirectiveType.STOP:
                        should_stop = True
                    elif d.directive_type == DirectiveType.RESUME:
                        should_start = True
                    else:
                        keep.append(d)

                # Keep non-control directives so they can be processed after start.
                for d in keep:
                    command_queue.push(d)

                if should_stop:
                    logger.info("STOP received before start. Exiting.")
                    close_run_log_session()
                    close_trajectory_session()
                    return
                if should_start:
                    agent_gate.set_state(AgentState.RUNNING)
                    console_log_silencer.disable()
                    logger.info("Chatapp start signal received. Beginning execution.")
                    break
    else:
        # Auto-start: set gate to running immediately
        from civStation.agent.modules.hitl.agent_gate import AgentState

        agent_gate.set_state(AgentState.RUNNING)
        console_log_silencer.disable()

    from civStation.utils.rich_logger import RichLogger

    rl = RichLogger.get()
    rl.start_live()

    logger.info(f"Starting execution for {args.turns} turn(s)...")
    try:
        runner_kwargs = {
            "router_provider": router_provider,
            "planner_provider": planner_provider,
            "normalizing_range": args.range,
            "delay_before_action": args.delay_action,
            "prompt_language": getattr(args, "prompt_language", "eng"),
            "high_level_strategy": args.strategy,
            "context_manager": ctx,
            "strategy_planner": strategy_planner,
            "knowledge_manager": knowledge_manager,
            "strategy_updater": strategy_updater,
            "turn_detector": turn_detector,
            "router_img_config": router_img_config,
            "planner_img_config": planner_img_config,
        }

        if args.turns == 1:
            run_one_turn(
                **runner_kwargs,
                macro_turn_manager=macro_turn_manager,
                state_bridge=state_bridge,
                context_updater=context_updater,
                debug_options=debug_options,
                command_queue=command_queue,
                agent_gate=agent_gate,
            )
        else:
            run_multi_turn(
                num_turns=args.turns,
                delay_between_turns=args.delay_turn,
                hitl_mode=args.hitl_mode,
                command_queue=command_queue,
                macro_turn_manager=macro_turn_manager,
                state_bridge=state_bridge,
                context_updater=context_updater,
                debug_options=debug_options,
                agent_gate=agent_gate,
                **runner_kwargs,
            )
    finally:
        rl.stop_live()
        if agent_gate:
            from civStation.agent.modules.hitl.agent_gate import AgentState

            agent_gate.set_state(AgentState.STOPPED)
        if turn_detector:
            turn_detector.stop()
        if strategy_updater:
            strategy_updater.stop()
        if context_updater:
            context_updater.stop()
        if queue_listener:
            queue_listener.stop()
        if status_server:
            status_server.stop()
        if relay_client:
            relay_client.stop()
        if chat_app:
            chat_app.stop()
            logger.info("Chat app stopped")
        close_run_log_session()
        close_trajectory_session()


if __name__ == "__main__":
    main()
