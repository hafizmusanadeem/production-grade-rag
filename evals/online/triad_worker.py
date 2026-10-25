"""Online triad scorer for LangSmith root runs."""
import hashlib
import os
import time
from datetime import datetime, timedelta, timezone

from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from evals.harness import run_evaluate
from evals.helper.deepeval_judge import get_judge

load_dotenv(override=True)
SAMPLE_RATE = 0.1
THRESHOLD = 0.7
POLL_SECONDS = 60
LAST_SCAN_TS = None


def _sampled(run):
    run_id = str(getattr(run, "id", "unknown"))
    return int(hashlib.sha256(run_id.encode()).hexdigest(), 16) % 10000 < SAMPLE_RATE * 10000


def _feedback_keys(client, run):
    return {feedback.key for feedback in client.list_feedback(run_ids=[run.id])}


def _has_score(client, run):
    return bool(_feedback_keys(client, run) & {"faithfulness", "answer_relevancy", "contextual_relevancy"})


def _messages_query(inputs):
    messages = inputs.get("messages", []) if isinstance(inputs, dict) else []
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return message.get("content", "")
    return inputs.get("query", "") if isinstance(inputs, dict) else ""


def _stratified_sample(run, feedback_keys):
    outputs = run.outputs or {}
    answer = outputs.get("final_answer", "")
    documents = outputs.get("documents", []) or []
    if not answer or not documents:
        return True
    if "user_rating" in feedback_keys:
        return True
    if "Kubernetes documentation assistant" in str(answer):
        return True
    return False


def score_recent_traces():
    global LAST_SCAN_TS
    from langsmith import Client
    client = Client()
    project = os.getenv("LANGSMITH_PROJECT", "rag-production")
    start_time = LAST_SCAN_TS - timedelta(minutes=2) if LAST_SCAN_TS else datetime.now(timezone.utc) - timedelta(days=1)
    runs = list(client.list_runs(project_name=project, is_root=True, run_type="chain", start_time=start_time))
    picked = declined = 0
    judge = get_judge()
    for run in runs:
        feedback = _feedback_keys(client, run)
        if {"faithfulness", "answer_relevancy", "contextual_relevancy"}.issubset(feedback):
            continue
        if not (_sampled(run) or _stratified_sample(run, feedback)):
            continue
        outputs = run.outputs or {}
        answer = outputs.get("final_answer", "")
        docs = outputs.get("documents", []) or []
        context = [str(doc).removeprefix("CONTENT: ") for doc in docs if isinstance(doc, str)]
        query = _messages_query(run.inputs or {})
        if not answer or not context:
            declined += 1
            continue
        cases = [
            ("faithfulness", FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True),
             dict(input=query, actual_output=answer, retrieval_context=context)),
            ("answer_relevancy", AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
             dict(input=query, actual_output=answer)),
            ("contextual_relevancy", ContextualRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
             dict(input=query, actual_output=answer, retrieval_context=context)),
        ]
        for key, metric, kwargs in cases:
            if key in feedback:
                continue
            try:
                run_evaluate(test_cases=[LLMTestCase(**kwargs)], metrics=[metric])
                client.create_feedback(run_id=run.id, key=key, score=metric.score, comment=metric.reason)
                picked += 1
            except Exception as exc:
                print(f"[{key}] failed on run {run.id}: {exc}")
    LAST_SCAN_TS = datetime.now(timezone.utc)
    print(f"[triad_worker] scored {picked} metrics; skipped {declined} declined/empty-context traces; scanned {len(runs)} runs")


if __name__ == "__main__":
    while True:
        score_recent_traces()
        time.sleep(POLL_SECONDS)
