"""Run:  python -m evals.helper.check_judge"""
import evals.helper.eval_env

from pydantic import BaseModel

from evals.helper.deepeval_judge import OpenRouterJudge


class Pong(BaseModel):
    answer: str


def main() -> None:
    judge = OpenRouterJudge()
    print("model :", judge.get_model_name())
    print("plain :", judge.generate("Reply with exactly: OK"))
    print("schema:", judge.generate('Return {"answer": "pong"}', schema=Pong))


if __name__ == "__main__":
    main()
