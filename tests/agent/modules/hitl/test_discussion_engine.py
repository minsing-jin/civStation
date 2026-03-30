import json

from fastapi.testclient import TestClient

from civStation.agent.modules.context.context_manager import ContextManager
from civStation.agent.modules.hitl.command_queue import CommandQueue
from civStation.agent.modules.hitl.status_ui.server import StatusServer
from civStation.agent.modules.hitl.status_ui.state_bridge import AgentStateBridge
from civStation.agent.modules.strategy.strategy_schemas import StructuredStrategy, VictoryType
from civStation.utils.chatapp.discussion import StrategyDiscussion
from civStation.utils.llm_provider.base import BaseVLMProvider, VLMResponse


class CapturingProvider(BaseVLMProvider):
    def __init__(self, responses: list[str]):
        super().__init__(api_key=None, model="capturing", resize_for_vlm=False)
        self.responses = list(responses)
        self.prompts: list[str] = []

    def _send_to_api(self, content_parts, temperature=0.7, max_tokens=8192, use_thinking=True) -> VLMResponse:
        text_parts = [part["text"] for part in content_parts if isinstance(part, dict) and "text" in part]
        self.prompts.append("\n".join(text_parts))
        if not self.responses:
            raise AssertionError("No more queued provider responses")
        return VLMResponse(content=self.responses.pop(0))

    def _build_image_content(self, image_path):
        return {"image_path": str(image_path)}

    def _build_pil_image_content(self, pil_image, jpeg_quality=None):
        return {"pil_size": getattr(pil_image, "size", None), "jpeg_quality": jpeg_quality}

    def _build_text_content(self, text: str):
        return {"text": text}

    def get_provider_name(self) -> str:
        return "capturing"


class TestDiscussionEngine:
    def setup_method(self):
        ContextManager.reset_instance()
        self.ctx = ContextManager.get_instance()
        self.ctx.update_global_context(current_turn=37, game_era="Medieval")
        self.ctx.set_strategy(
            StructuredStrategy(
                text="캠퍼스 > 시장 > 대학 순으로 가고, 골드 부족을 먼저 안정화한다.",
                victory_goal=VictoryType.SCIENCE,
                current_phase="mid_development",
            )
        )
        self.ctx.high_level_context.add_threat("북쪽 국경 방어가 약함")
        self.ctx.high_level_context.add_opportunity("산 인접 캠퍼스 부지 확보 가능")

    def teardown_method(self):
        ContextManager.reset_instance()

    def test_process_message_uses_explicit_strategy_and_context_snapshot(self):
        provider = CapturingProvider(["좋습니다. 골드 안정화 후 캠퍼스 타이밍을 맞추세요."])
        discussion = StrategyDiscussion(vlm_provider=provider, context_manager=self.ctx)

        session_id = discussion.create_session("tester")
        response = discussion.process_message(session_id, "지금은 뭘 우선해야 해?", language="ko")

        assert "골드 안정화" in response
        prompt = provider.prompts[0]
        assert "현재 High-Level Strategy:" in prompt
        assert self.ctx.get_strategy_string() in prompt
        assert "현재 게임 Context:" in prompt
        assert self.ctx.get_combined_context() in prompt

    def test_finalize_session_uses_strategy_and_context_snapshot(self):
        finalized_json = json.dumps(
            {
                "victory_goal": "science",
                "current_phase": "mid_development",
                "text": "골드 안정화 > 캠퍼스 > 대학 순으로 정리하고 북쪽 방어를 유지한다.",
            }
        )
        provider = CapturingProvider([finalized_json])
        discussion = StrategyDiscussion(vlm_provider=provider, context_manager=self.ctx)

        session_id = discussion.create_session("tester")
        discussion._sessions[session_id].add_user_message("과학 승리 유지하되 경제를 먼저 안정화하자.")
        strategy = discussion.finalize_session(session_id)

        assert strategy is not None
        assert strategy.text.startswith("골드 안정화")
        prompt = provider.prompts[0]
        assert "현재 High-Level Strategy:" in prompt
        assert "현재 게임 Context:" in prompt

    def test_discuss_status_exposes_current_strategy_and_context(self):
        provider = CapturingProvider(["unused"])
        discussion = StrategyDiscussion(vlm_provider=provider, context_manager=self.ctx)
        bridge = AgentStateBridge(self.ctx, CommandQueue())
        app = StatusServer(bridge, CommandQueue(), discussion_engine=discussion)._create_app()

        with TestClient(app) as client:
            response = client.get("/api/discuss/status")

        assert response.status_code == 200
        data = response.json()
        assert data["active"] is False
        assert data["current_strategy"] == self.ctx.get_strategy_string()
        assert data["combined_context"] == self.ctx.get_combined_context()
