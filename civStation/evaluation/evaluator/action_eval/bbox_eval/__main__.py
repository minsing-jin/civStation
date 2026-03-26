"""Allow running as: python -m civStation.evaluation.evaluator.action_eval.bbox_eval"""

import sys

from .cli import main

sys.exit(main())
