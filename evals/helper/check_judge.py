"""evals/helper/check_judge.py -- sanity-check the OpenRouterJudge wiring."""
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.helper.deepeval_judge import OpenRouterJudge

try:
    import os
    if not os.getenv("OPENROUTER_API_KEY"):
        print("WARN: OPENROUTER_API_KEY not set -- judge will fail at runtime.")
    judge = OpenRouterJudge()
    print(f"OpenRouterJudge OK: model={judge.get_model_name()}")
except Exception as exc:
    print(f"check_judge FAILED: {exc}")
    sys.exit(1)
