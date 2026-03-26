from civStation.agent.models.schema import AgentPlan, ClickAction, KeyPressAction
from civStation.agent.modules.primitive.base_primitive import BasePrimitive
from civStation.agent.modules.router.router import Civ6Router
from civStation.evaluation.evaluator.action_eval.base_static_primitive_evaluator import (
    BaseEvaluator,
    EvalResult,
    GroundTruth,
)

# --- Civ6 Primitives implementation example (Civ6 Primitives 구현 예시) ---


class Civ6Primitive(BasePrimitive):
    def __init__(self, name):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def generate_plan_and_action(self, screenshot_path: str) -> AgentPlan:
        # [TODO] Real VLM call goes here (mock return for now)
        # (실제 VLM 호출 부분 — 여기서는 Mock Return)

        # Example: assume a unit-move command (예시: 유닛 이동 명령)
        return AgentPlan(
            primitive_name=self.name,
            reasoning="Settler needs to move to fresh water.",
            actions=[
                ClickAction(x=1920 // 2, y=1080 // 2, description="Select Unit"),
                KeyPressAction(keys=["m"], description="Move Command"),
                ClickAction(x=1000, y=500, description="Target Tile"),
            ],
        )


# --- Civ6 Evaluator (includes comparison logic / 비교 로직 포함) ---


class Civ6StaticEvaluator(BaseEvaluator):
    def _compare(self, gt: GroundTruth, selected_prim: str, plan: AgentPlan) -> EvalResult:
        # 1. Primitive selection accuracy (Primitive 선택 정확도)
        prim_match = gt.expected_primitive == selected_prim

        # 2. Action sequence accuracy — simple comparison example
        # (실제로는 좌표 허용 오차(Tolerance)나 순서 유사도를 계산해야 함)
        actions_match = True
        if len(gt.expected_actions) != len(plan.actions):
            actions_match = False
        else:
            for gt_act, pred_act in zip(gt.expected_actions, plan.actions, strict=False):
                # Pydantic model comparison — type and value must match
                # (좌표의 경우 ±10px 여유를 주는 로직 추가 가능)
                if gt_act.model_dump(exclude={"description"}) != pred_act.model_dump(exclude={"description"}):
                    actions_match = False
                    break

        return EvalResult(
            screenshot_path=gt.screenshot_path,
            selected_primitive=selected_prim,
            primitive_match=prim_match,
            action_sequence_match=actions_match,
            levenshtein_distance=0,  # TODO: apply edit-distance algorithm (편집 거리 알고리즘 적용)
        )


# --- Example runner (실행 예시) ---


def run_civ6_evaluation():
    # 1. Initialize primitives (Primitives 초기화)
    primitives = {
        "unit_ops_primitive": Civ6Primitive("unit_ops_primitive"),
        "country_mayer_primitive": Civ6Primitive("country_mayer_primitive"),
        "research_select_primitive": Civ6Primitive("research_select_primitive"),
        "culture_decision_primitive": Civ6Primitive("culture_decision_primitive"),
    }

    # 2. Create evaluation pipeline (파이프라인 생성)
    router = Civ6Router()
    evaluator = Civ6StaticEvaluator(router, primitives)

    # 3. Load test set — in practice, loaded from JSON (실제로는 JSON 파일에서 로드)
    test_set = [
        GroundTruth(
            screenshot_path="civ6_unit_move_01.png",
            expected_primitive="unit_ops_primitive",
            expected_actions=[ClickAction(x=960, y=540), KeyPressAction(keys=["m"]), ClickAction(x=1000, y=500)],
        )
    ]

    # 4. Evaluation loop (루프 실행)
    results = []
    for case in test_set:
        print(f"Evaluating {case.screenshot_path}...")
        res = evaluator.evaluate_single(case)
        results.append(res)

    # 5. Compute final metrics (최종 Metric 계산)
    accuracy = sum([r.primitive_match and r.action_sequence_match for r in results]) / len(results)
    print(f"Final Accuracy: {accuracy * 100}%")


if __name__ == "__main__":
    run_civ6_evaluation()
