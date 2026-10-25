"""Sanity-check shared judge wiring."""
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv(override=True)
from evals.helper.deepeval_judge import get_judge


def main():
    try:
        judge = get_judge()
        print(f"OpenRouterJudge OK: model={judge.get_model_name()}")
        return 0
    except Exception as exc:
        print(f"check_judge FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
