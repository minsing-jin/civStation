# src/agent/base.py (개념 코드)
class BaseAgent:
    def __init__(self, router, action_generator, image_processor):
        self.router = router  # Primitive Router
        self.action_generator = action_generator  # Primitive Action Generator
        self.image_processor = image_processor  # 이미지 전처리 모듈

    def step(self, screenshot, instruction):
        # 1. Load Screenshot & Preprocess
        state = self.process_image(screenshot)

        # 2. Primitive Router (어떤 종류의 행동을 할지 결정)
        intent = self.router.predict(state, instruction)

        # 3. Primitive Action (구체적인 행동 파라미터 결정)
        action = self.action_generator.generate(state, intent)

        return {"intent": intent, "action": action}
