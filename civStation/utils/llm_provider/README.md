# 🤖 VLM Provider Integration Guide

## 📚 Index

- [🔍 Overview](#-overview)
- [🚀 Quick Start](#-quick-start)
- [🧩 Provider Details](#-provider-details)
- [💻 Programmatic Usage](#-programmatic-usage)
- [🏗️ Provider Architecture](#-provider-architecture)

This guide explains how to use Vision-Language Model (VLM) providers with the Civilization VI Static Primitive Evaluator.

## 🔍 Overview

The evaluation pipeline supports multiple VLM providers:
- **Claude** (Anthropic): Claude 3.5 Sonnet, Opus, Haiku
- **Gemini** (Google): Gemini 2.0 Flash, Pro
- **GPT** (OpenAI): GPT-4o, GPT-4o-mini, GPT-4 Vision
- **Mock**: Deterministic mocking for testing (no API calls)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# For Claude
pip install anthropic

# For Gemini
pip install google-generativeai Pillow

# For GPT
pip install openai
```

### 2. Set API Keys

```bash
# Claude
export ANTHROPIC_API_KEY="your-api-key"

# Gemini
export GENAI_API_KEY="your-api-key"

# GPT
export OPENAI_API_KEY="your-api-key"
```

### 3. Run the Agent

```bash
# Recommended CLI
uv run civstation run --provider gemini --model gemini-3-flash --turns 10

# Legacy module entrypoint still works
python -m civStation.agent.turn_runner --provider gemini --model gemini-3-flash --turns 10
```

## 🧩 Provider Details

### Claude (Anthropic)

**Default Model**: `claude-3-5-sonnet-20241022`

**Supported Models**:
- `claude-3-5-sonnet-20241022` (Recommended)
- `claude-3-opus-20240229`
- `claude-3-haiku-20240307`

**Pricing** (approximate, as of 2024):
- Opus: $15/$75 per million input/output tokens
- Sonnet: $3/$15 per million tokens
- Haiku: $0.25/$1.25 per million tokens

**Usage**:
```python
from civStation.utils.provider import create_provider

provider = create_provider("claude", model="claude-3-5-sonnet-20241022")
plan = provider.call_and_parse(
    prompt="Analyze this Civ6 screenshot...",
    image_path="screenshot.png",
    primitive_name="unit_ops_primitive"
)
```

### Gemini (Google)

**Default Model**: `gemini-2.0-flash-exp`

**Supported Models**:
- `gemini-2.0-flash-exp` (Fast, recommended)
- `gemini-1.5-pro`
- `gemini-pro-vision`

**Pricing** (approximate):
- Flash: $0.075 per million tokens
- Pro: $0.125 per million tokens

**Usage**:
```python
provider = create_provider("gemini", model="gemini-2.0-flash-exp")
plan = provider.call_and_parse(
    prompt="Analyze this Civ6 screenshot...",
    image_path="screenshot.png",
    primitive_name="science_decision_primitive"
)
```

### GPT (OpenAI)

**Default Model**: `gpt-4o`

**Supported Models**:
- `gpt-4o` (Recommended)
- `gpt-4o-mini` (Cheaper)
- `gpt-4-turbo`
- `gpt-4-vision-preview`

**Pricing** (approximate):
- GPT-4o: $2.5/$10 per million input/output tokens
- GPT-4o-mini: $0.15/$0.6 per million tokens (cheapest!)
- GPT-4 Turbo: $10/$30 per million tokens

**Usage**:
```python
provider = create_provider("gpt", model="gpt-4o-mini")
plan = provider.call_and_parse(
    prompt="Analyze this Civ6 screenshot...",
    image_path="screenshot.png",
    primitive_name="culture_decision_primitive"
)
```

## 💻 Programmatic Usage

### Using Providers in Code

```python
from civStation.utils.provider import create_provider

# Create provider
provider = create_provider(
    provider_name="claude",
    api_key="your-api-key",  # Optional if using env var
    model="claude-3-5-sonnet-20241022"  # Optional, uses default
)

# Call VLM
response = provider.call_vlm(
    prompt="What do you see in this image?",
    image_path="screenshot.png",
    temperature=0.7,
    max_tokens=4096
)

print(f"Response: {response.content}")
print(f"Tokens: {response.tokens_used}")
print(f"Cost: ${response.cost:.4f}")

# Parse to AgentPlan
plan = provider.parse_to_agent_plan(response, "unit_ops_primitive")
print(f"Reasoning: {plan.reasoning}")
print(f"Actions: {len(plan.actions)}")
```

### Using with Primitives

```python

from civStation.agent.modules.primitive.primitives import UnitOpsPrimitive
from civStation.utils.provider import create_provider

# Create provider
provider = create_provider("gpt", model="gpt-4o-mini")

# Create primitive with provider
primitive = UnitOpsPrimitive(vlm_provider=provider)

# Generate plan (will use VLM)
plan = primitive.generate_plan_and_action("screenshot.png")

print(f"Generated {len(plan.actions)} actions")
for i, action in enumerate(plan.actions, 1):
    print(f"  {i}. {action.type}: {action}")
```

### Factory Function

```python
from civStation.utils.provider import create_provider, get_available_providers

# List available providers
providers = get_available_providers()
print(f"Available: {list(providers.keys())}")
print(f"Defaults: {providers}")

# Create by name
claude = create_provider("claude")
gemini = create_provider("gemini")
gpt = create_provider("gpt")
mock = create_provider("mock")  # No API key needed
```

## 🏗️ Provider Architecture

Each provider implements the `BaseVLMProvider` interface:

```python
class BaseVLMProvider(ABC):
    def call_vlm(self, prompt, image_path, temperature, max_tokens) -> VLMResponse:
        """Call the VLM API"""
        pass

    def parse_to_agent_plan(self, response, primitive_name) -> AgentPlan:
        """Parse response into actions"""
        pass

    def call_and_parse(self, prompt, image_path, primitive_name) -> AgentPlan:
        """Convenience: call and parse in one step"""
        pass
```

### Response Format

All providers expect JSON responses in this format:

```json
{
  "reasoning": "Brief explanation of the analysis and chosen actions",
  "actions": [
    {
      "type": "click",
      "x": 100,
      "y": 200,
      "description": "Optional action description"
    },
    {
      "type": "press",
      "keys": ["m"],
      "description": "Move command"
    },
    {
      "type": "drag",
      "start_x": 10,
      "start_y": 20,
      "end_x": 100,
      "end_y": 200,
      "duration": 0.5
    }
  ]
}
```

## Primitive Integration

Primitives accept an optional `vlm_provider` parameter:

```python
# Without provider (uses mock)
primitive = UnitOpsPrimitive()
plan = primitive.generate_plan_and_action("screenshot.png")  # Mock actions

# With provider (uses real VLM)
provider = create_provider("claude")
primitive = UnitOpsPrimitive(vlm_provider=provider)
plan = primitive.generate_plan_and_action("screenshot.png")  # Real VLM call
```

## Cost Estimation

The providers automatically estimate costs:

```python
provider = create_provider("gpt", model="gpt-4o-mini")
response = provider.call_vlm(
    prompt="Analyze screenshot",
    image_path="test.png"
)

print(f"Tokens: {response.tokens_used}")
print(f"Estimated cost: ${response.cost:.4f}")
```

**Typical costs per evaluation** (6 test cases):
- Mock: $0.00 (FREE)
- GPT-4o-mini: ~$0.01 - $0.05
- Gemini Flash: ~$0.01 - $0.03
- Claude Sonnet: ~$0.10 - $0.30
- Claude Opus: ~$0.50 - $1.50
- GPT-4o: ~$0.20 - $0.60

## Error Handling

All providers handle errors gracefully:

```python
try:
    provider = create_provider("claude")
    plan = provider.call_and_parse(
        prompt="Analyze...",
        image_path="screenshot.png",
        primitive_name="unit_ops"
    )
except ValueError as e:
    print(f"Configuration error: {e}")
except RuntimeError as e:
    print(f"API call failed: {e}")
except FileNotFoundError as e:
    print(f"Image not found: {e}")
```

## Best Practices

1. **Use Mock for Development**: Free and fast
2. **Use GPT-4o-mini for Testing**: Cheapest real VLM
3. **Use Claude Sonnet for Production**: Best balance of quality/cost
4. **Set Temperature Low (0.3-0.5)**: More deterministic for actions
5. **Cache API Keys in Environment**: More secure than hardcoding
6. **Monitor Costs**: Check `response.cost` regularly
7. **Handle API Failures**: Implement retry logic for production

## Testing

```bash
pytest
```

For live model runs, use the normal agent entrypoints above.

## Troubleshooting

### "API key must be provided"
Set environment variable or pass `api_key` parameter:
```bash
export ANTHROPIC_API_KEY="your-key"
# or
python script.py --api-key your-key
```

### "Failed to parse response"
The VLM returned invalid JSON. Check the prompt or use a more capable model.

### "Image not found"
Verify the screenshot path is correct and file exists.

### Import errors
Install the provider's SDK:
```bash
pip install anthropic  # For Claude
pip install google-generativeai Pillow  # For Gemini
pip install openai  # For GPT
```

## Where VLM is Used

**실제 모델이 사용되는 위치:**

1. **Primitive Classes** (`civ6_impl.py`):
   ```python
   # civ6_impl.py lines 46-92
   def generate_plan(self, screenshot_path: str) -> AgentPlan:
       if self.vlm_provider:  # ← VLM 사용하는 부분
           return self.vlm_provider.call_and_parse(
               prompt=prompt,
               image_path=screenshot_path,
               primitive_name=self.name,
           )
   ```

2. **Main Evaluation Loop** (`main.py`):
   ```python
   # main.py lines 146-165
   primitives = {
       "unit_ops_primitive": UnitOpsPrimitive(vlm_provider=provider),  # ← VLM 주입
       ...
   }
   ```

3. **VLM Providers** (`provider/*.py`):
   ```python
   # claude.py, gemini.py, gpt.py
   def call_vlm(self, prompt, image_path, ...) -> VLMResponse:
       response = self.client.messages.create(...)  # ← 실제 API 호출
       return VLMResponse(content=response_text, ...)
   ```

**흐름:**
1. `main.py`가 VLM provider 생성 (optional)
2. Primitive에 provider 주입
3. `primitive.generate_plan()` 호출시 VLM이 있으면 사용, 없으면 mock
4. VLM이 screenshot 분석 → JSON 응답 → AgentPlan 파싱
5. Evaluator가 ground truth와 비교

## Contributing

To add a new provider:
1. Create `civStation/utils/provider/your_provider.py`
2. Implement `BaseVLMProvider` interface
3. Add to `__init__.py` factory
4. Update this README
