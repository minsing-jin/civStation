# Screenshot Fixtures

This directory contains screenshot files for integration testing of the Civ6 Static Primitive Evaluator.

## Directory Structure

```
screenshots/
├── README.md                    # This file
├── turn_10_science.png         # Science decision screenshot (if available)
├── unit_settler_move.png       # Unit movement screenshot (if available)
├── city_mayor_production.png   # City management screenshot (if available)
├── culture_civic_choice.png    # Culture decision screenshot (if available)
├── unit_warrior_combat.png     # Combat screenshot (if available)
└── science_tech_boost.png      # Tech boost screenshot (if available)
```

## How to Add Screenshots

To run integration tests with real screenshots:

1. **Capture Screenshots**: Take screenshots from Civilization VI gameplay at relevant moments:
   - Science tree decisions
   - Unit operations (movement, combat)
   - City management screens
   - Culture/civics choices

2. **Name Files**: Use descriptive names that match the ground truth data in `../ground_truth.json`:
   - `turn_10_science.png` - Science decision screen
   - `unit_settler_move.png` - Unit ready to move
   - `city_mayor_production.png` - City production screen
   - `culture_civic_choice.png` - Civics selection screen
   - `unit_warrior_combat.png` - Unit in combat
   - `science_tech_boost.png` - Tech tree with boost notification

3. **Place Files**: Copy your screenshot files to this directory

4. **Update Ground Truth**: If needed, update `../ground_truth.json` with the correct actions for your screenshots

## Expected Screenshot Characteristics

- **Format**: PNG or JPEG
- **Resolution**: Typically 1920x1080 (Full HD) or 2560x1440 (2K)
- **Content**: Clear view of the relevant game screen element
- **File Size**: Typically 100KB - 5MB depending on resolution

## Testing Without Screenshots

The integration tests will automatically skip if screenshot files are not present. The evaluator can still be tested using the unit tests in:
- `test_tolerance.py` - Coordinate tolerance tests
- `test_discriminated_union.py` - Action parsing tests

## Ground Truth Format

Each screenshot should have corresponding ground truth in `../ground_truth.json`:

```json
{
  "screenshot": "filename.png",
  "gt_primitive": "primitive_name",
  "gt_actions": [
    {"type": "click", "x": 100, "y": 200},
    {"type": "press", "keys": ["esc"]}
  ]
}
```

## Primitive Types

Available primitives:
- `unit_ops_primitive` - Unit operations (movement, combat, fortify)
- `country_mayer_primitive` - City management (production, citizens)
- `science_decision_primitive` - Technology tree decisions
- `culture_decision_primitive` - Civics tree decisions

## Action Types

Supported action types:
- `click` - Mouse click at coordinates
- `press` - Keyboard key press
- `drag` - Mouse drag from start to end coordinates
- `wait` - Wait/delay action
- `double_click` - Double click at coordinates

## Running Integration Tests

```bash
# Run all tests including integration tests (will skip if no screenshots)
pytest tests/evaluator/static_eval/civ6_eval/test_evaluation_integration.py

# Run only integration tests
pytest tests/evaluator/static_eval/civ6_eval/test_evaluation_integration.py -m integration

# Run with verbose output
pytest tests/evaluator/static_eval/civ6_eval/test_evaluation_integration.py -v
```

## Troubleshooting

**Tests are skipping**: Screenshots are not present in this directory. Add screenshot files to enable integration tests.

**Tests are failing**:
1. Verify screenshot filenames match `ground_truth.json`
2. Check that ground truth actions are correct for your screenshots
3. Ensure screenshot files are valid image files
4. Verify file permissions allow reading

## Privacy Note

Do not commit copyrighted game screenshots to public repositories. This directory should remain empty in version control, with screenshots added locally for testing purposes only.