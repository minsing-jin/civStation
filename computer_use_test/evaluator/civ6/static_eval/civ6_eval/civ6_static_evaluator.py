from computer_use_test.evaluator.civ6.static_eval.base_static_primitive_evaluator import BaseEvaluator, GroundTruth, \
    EvalResult, BasePrimitive, PrimitiveRouter
from computer_use_test.agent.models.civ6_models import ClickAction, KeyPressAction, AgentPlan


# --- Civ6 Primitives 구현 ---

class Civ6Primitive(BasePrimitive):
    def __init__(self, name):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def generate_plan(self, screenshot_path: str) -> AgentPlan:
        # [TODO] 실제 VLM 호출 부분 (여기서는 Mock Return)
        # prompt = f"Analyze {screenshot_path} for {self.name}..."
        # response = call_vlm(prompt)

        # 예시: 유닛 이동 명령이라고 가정
        return AgentPlan(
            primitive_name=self.name,
            reasoning="Settler needs to move to fresh water.",
            actions=[
                ClickAction(x=1920 // 2, y=1080 // 2, description="Select Unit"),
                KeyPressAction(keys=['m'], description="Move Command"),
                ClickAction(x=1000, y=500, description="Target Tile")
            ]
        )


# --- Civ6 Router 구현 ---

class Civ6Router(PrimitiveRouter):
    def route(self, screenshot_path: str) -> str:
        # [TODO] VLM에게 스크린샷을 주고 Primitive 분류 요청
        # 여기서는 파일명에 힌트가 있다고 가정하고 Mocking
        if "unit" in screenshot_path: return "unit_ops_primitive"
        if "mayor" in screenshot_path: return "country_mayer_primitive"
        if "science" in screenshot_path: return "science_decision_primitive"
        return "culture_decision_primitive"


# --- Civ6 Evaluator (비교 로직 포함) ---

class Civ6StaticEvaluator(BaseEvaluator):
    def _compare(self, gt: GroundTruth, selected_prim: str, plan: AgentPlan) -> EvalResult:
        # 1. Primitive 선택 정확도
        prim_match = (gt.expected_primitive == selected_prim)

        # 2. Action Sequence 정확도 (단순 비교 예시)
        # 실제로는 좌표의 허용 오차(Tolerance)나 순서의 유사도를 계산해야 함
        actions_match = True
        if len(gt.expected_actions) != len(plan.actions):
            actions_match = False
        else:
            for gt_act, pred_act in zip(gt.expected_actions, plan.actions):
                # Pydantic 모델 비교 (타입과 값이 모두 같아야 함)
                # 좌표의 경우 ±10 픽셀 정도 여유를 주는 로직 추가 가능
                if gt_act.model_dump(exclude={'description'}) != pred_act.model_dump(exclude={'description'}):
                    actions_match = False
                    break

        return EvalResult(
            screenshot_path=gt.screenshot_path,
            selected_primitive=selected_prim,
            primitive_match=prim_match,
            action_sequence_match=actions_match,
            levenshtein_distance=0  # 추후 구현: 편집 거리 알고리즘 적용
        )


# --- 실행 예시 (Main Loop) ---

def run_civ6_evaluation():
    # 1. Primitives 초기화
    primitives = {
        "unit_ops_primitive": Civ6Primitive("unit_ops_primitive"),
        "country_mayer_primitive": Civ6Primitive("country_mayer_primitive"),
        "science_decision_primitive": Civ6Primitive("science_decision_primitive"),
        "culture_decision_primitive": Civ6Primitive("culture_decision_primitive"),
    }

    # 2. 파이프라인 생성
    router = Civ6Router()
    evaluator = Civ6StaticEvaluator(router, primitives)

    # 3. Test Set (Ground Truth) 로드 - 실제로는 JSON 파일에서 로드
    test_set = [
        GroundTruth(
            screenshot_path="civ6_unit_move_01.png",
            expected_primitive="unit_ops_primitive",
            expected_actions=[
                ClickAction(x=960, y=540),
                KeyPressAction(keys=['m']),
                ClickAction(x=1000, y=500)
            ]
        )
    ]

    # 4. 루프 실행 (다이어그램의 loop)
    results = []
    for case in test_set:
        print(f"Evaluating {case.screenshot_path}...")
        res = evaluator.evaluate_single(case)
        results.append(res)

    # 5. 최종 Metric 계산
    accuracy = sum([r.primitive_match and r.action_sequence_match for r in results]) / len(results)
    print(f"Final Accuracy: {accuracy * 100}%")


if __name__ == "__main__":
    run_civ6_evaluation()
