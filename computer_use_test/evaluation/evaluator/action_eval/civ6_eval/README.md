# Civilization VI Static Primitive Evaluator

A comprehensive evaluation pipeline for assessing VLM-based AI agents playing Civilization VI. The system evaluates both **Primitive Selection** (strategy choice) and **Action Generation** (PyAutoGUI command sequences).

## Overview

This evaluator implements a static testing framework that:
1. Loads ground truth test cases from JSON
2. Routes screenshots to appropriate primitives (strategies)
3. Generates action sequences for each test case
4. Compares predictions against ground truth with configurable tolerances
5. Reports detailed accuracy metrics

## Architecture

### Core Components

#### 1. Schema (`schema.py`)
Defines Pydantic models with **discriminated union** for type-safe action serialization:
- `ClickAction`: Mouse click at coordinates
- `KeyPressAction`: Keyboard input
- `DragAction`: Mouse drag from start to end coordinates
- `AgentPlan`: Complete action sequence with reasoning

#### 2. Interfaces (`interfaces.py`)
Abstract base classes and utilities:
- `BasePrimitive`: Strategy pattern for game scenarios
- `PrimitiveRouter`: Routes screenshots to primitives
- `BaseEvaluator`: Evaluation pipeline orchestrator
- `within_tolerance()`: Helper for coordinate comparison

#### 3. Implementation (`civ6_impl.py`)
Concrete implementations for Civ6:

**Four Primitives:**
- `UnitOpsPrimitive`: Unit movement, combat, fortification
- `CountryMayerPrimitive`: City management, production
- `ScienceDecisionPrimitive`: Technology tree decisions
- `CultureDecisionPrimitive`: Civics tree decisions

**Router:**
- `Civ6MockRouter`: Keyword-based primitive selection (for testing)

**Evaluator:**
- `Civ6StaticEvaluator`: Comparison logic with 5-pixel coordinate tolerance

#### 4. Runner (`main.py`)
Main execution script with JSON loading and metrics reporting.

## Usage

### Basic Usage

```bash
# From project root
python -m computer_use_test.evaluator.static_eval.civ6_eval.main

# With custom test file
python -m computer_use_test.evaluator.static_eval.civ6_eval.main path/to/test_set.json
```

### Test Data Format

Create a JSON file with test cases:

```json
[
  {
    "screenshot": "turn_10_science.png",
    "gt_primitive": "science_decision_primitive",
    "gt_actions": [
      {"type": "click", "x": 100, "y": 200},
      {"type": "press", "keys": ["esc"]}
    ]
  },
  {
    "screenshot": "unit_settler_move.png",
    "gt_primitive": "unit_ops_primitive",
    "gt_actions": [
      {"type": "click", "x": 960, "y": 540},
      {"type": "press", "keys": ["m"]},
      {"type": "click", "x": 1000, "y": 500}
    ]
  }
]
```

### Action Types

**Click Action:**
```json
{
  "type": "click",
  "x": 100,
  "y": 200,
  "button": "left"  // optional, defaults to "left"
}
```

**Key Press Action:**
```json
{
  "type": "press",
  "keys": ["esc"],
  "interval": 0.1  // optional, defaults to 0.1
}
```

**Drag Action:**
```json
{
  "type": "drag",
  "start_x": 100,
  "start_y": 100,
  "end_x": 200,
  "end_y": 200,
  "duration": 0.5,  // optional, defaults to 0.5
  "button": "left"   // optional, defaults to "left"
}
```

## Evaluation Logic

### Primitive Selection
- **Exact match required**: Predicted primitive name must match ground truth

### Action Sequence Comparison
1. **Length check**: Must have same number of actions
2. **Sequential comparison**: Each action compared in order
3. **Coordinate tolerance**: ±5 pixels for x/y coordinates (configurable)
4. **Key matching**: Exact match required for keyboard keys

### Metrics Reported
- **Primitive Selection Accuracy**: % of correct primitive choices
- **Action Sequence Accuracy**: % of perfectly matched action sequences
- **Overall Accuracy**: % where both primitive AND actions are correct

## Testing

### Run All Validations

```bash
# Test 5-pixel tolerance
python -m computer_use_test.evaluator.static_eval.civ6_eval.test_tolerance

# Test discriminated union
python -m computer_use_test.evaluator.static_eval.civ6_eval.test_discriminated_union

# Run full pipeline
python -m computer_use_test.evaluator.static_eval.civ6_eval.main
```

### Expected Output

```
============================================================
Civilization VI Static Primitive Evaluator
============================================================

Loaded 6 test cases from test_set.json

Running evaluation...

[1/6] turn_10_science.png
  Primitive: ✓ (Expected: science_decision_primitive, Got: science_decision_primitive)
  Actions:   ✓ (2 actions)

...

============================================================
Final Metrics
============================================================
Primitive Selection Accuracy: 100.00%  (6/6)
Action Sequence Accuracy:      83.33%  (5/6)
Overall Accuracy (Both):       83.33%  (5/6)

✓ Good performance. Most test cases passed.
```

## Customization

### Adjust Coordinate Tolerance

Edit `civ6_impl.py`:

```python
class Civ6StaticEvaluator(BaseEvaluator):
    COORD_TOLERANCE = 10  # Change from 5 to 10 pixels
```

### Add New Primitives

1. Create a new primitive class:
```python
class NewPrimitive(BasePrimitive):
    @property
    def name(self) -> str:
        return "new_primitive"

    def generate_plan(self, screenshot_path: str) -> AgentPlan:
        # Implementation
        pass
```

2. Register in `main.py`:
```python
primitives = {
    "unit_ops_primitive": UnitOpsPrimitive(),
    "new_primitive": NewPrimitive(),
    # ...
}
```

3. Update router in `civ6_impl.py`:
```python
if "new_keyword" in filename:
    return "new_primitive"
```

## Files Overview

```
civ6_eval/
├── README.md                      # This file
├── civ6_impl.py                   # Primitives, router, evaluator
├── main.py                        # Main runner script
├── test_set.json                  # Sample test data
├── test_tolerance.py              # Tolerance validation tests
└── test_discriminated_union.py   # Union validation tests
```

## Integration with VLM

The current implementation uses **mock** primitives for testing. To integrate with a real VLM:

1. Replace `generate_plan()` in each primitive:
```python
def generate_plan(self, screenshot_path: str) -> AgentPlan:
    # Call your VLM
    prompt = f"Analyze {screenshot_path} for {self.name}..."
    response = call_vlm(prompt)  # Your VLM integration

    # Parse response into actions
    actions = parse_vlm_response(response)

    return AgentPlan(
        primitive_name=self.name,
        reasoning=response.reasoning,
        actions=actions
    )
```

2. Replace `Civ6MockRouter.route()`:
```python
def route(self, screenshot_path: str) -> str:
    # Ask VLM to classify the screenshot
    prompt = f"What game scenario is shown in {screenshot_path}?"
    response = call_vlm_classifier(prompt)
    return response.primitive_name
```

## Troubleshooting

### ModuleNotFoundError
Run from project root using module syntax:
```bash
python -m computer_use_test.evaluator.static_eval.civ6_eval.main
```

### JSON Parse Errors
Ensure your JSON follows the exact schema with `"type"` as discriminator field.

### All Actions Failing
Check that:
1. Mock primitives are generating actions with correct coordinates
2. Tolerance is sufficient for your use case
3. Action types match exactly (ClickAction vs KeyPressAction)

## Future Enhancements

- [ ] Implement Levenshtein distance for partial credit
- [ ] Add visualization of predicted vs ground truth actions
- [ ] Support for additional action types (scroll, right-click, etc.)
- [ ] Batch processing with parallel evaluation
- [ ] Export results to CSV/JSON for analysis
- [ ] Integration with CI/CD for regression testing