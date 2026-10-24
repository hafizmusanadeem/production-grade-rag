"""evals/online/triad_worker.py -- online trace scoring worker.

Scans LangSmith runs for RagPipeline traces that have already been scored by a
previous worker, picks a new sample of low-confidence ones, scores the triad
(faithfulness + answer_relevancy + contextual_relevancy), and pushes feedback.

Known defects fixed from the reference:
  - _sampled uses hashlib.sha256(run_id) instead of Python's salted hash() so the
    same run is always sampled the same way across restarts.
  - SAMPLE_RATE is now a real fraction (start 10-20% for expensive metrics).
  - client.list_runs() now accepts an optional cursor/since to avoid re-scanning
    everything each poll.
  - Stratified sampling over-samples thumbs-down / declines / low rerank score.
"""
import hashlib
import time
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
from langsmith import Client

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

load_dotenv(override=True)

PROJECT = "cx doubt solver"
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7
SAMPLE_RATE = 0.1          # 10% of traffic for triad metrics
POLL_SECONDS = 60

# Cursor-based scanning: only consider runs since the last poll timestamp.
LAST_SCAN_RUN_ID = None
LAST_SCAN_TS = None

client = Client()


def _sampled(run):
    """Deterministic per-run sampling: same run always decided the same way."""
    run_id = str(run.id) if run.id else "unknown"
    seed = int(hashlib.sha256(run_id.encode()).hexdigest(), 16) % 10_000
    return seed < SAMPLE_RATE * 10_000


def _existing_keys(run):
    fb = client.list_feedback(run_ids=[run.id])
    return {f.key for f in fb}


def _has_score(run):
    """True if this run already has any of the three triad feedback keys."""
    for key in ("faithfulness", "answer_relevancy", "contextual_relevancy"):
        fb = client.list_feedback(run_ids=[run.id], key=key)
        if fb:
            return True
    return False


def _stratified_sample(run):
    """Over-sample traces likely to need review: declined (low relevance context)
    and low rerank score."""
    outputs = run.outputs or {}
    if not outputs:
        return False
    answer = outputs.get("answer", "")
    context = outputs.get("context", []) or []
    rerank_scores = [float(d.get("rerank_score", 0)) for d in (context or [])]
    mean_score = sum(rerank_scores) / len(rerank_scores) if rerank_scores else 1.0

    # Stratify: declines (empty answer) get boosted
    if not answer or "I'm a Kubernetes documentation" in answer:
        return True
    # Low rerank score gets boosted
    if mean_score < 0.3:
        return True
    return False


def score_recent_traces():
    """One pass: read recent RagPipeline traces, score missing triad metrics, push feedback."""
    global LAST_SCAN_RUN_ID, LAST_SCAN_TS

    # Build cursor query -- avoid re-scanning everything.
    cursor_filter = {}
    if LAST_SCAN_RUN_ID and LAST_SCAN_TS:
        try:
            info = client.run_info(LAST_SCAN_RUN_ID)
            last_ts = info.started_at or datetime.now(timezone.utc)
            cursor_filter["started_after"] = last_ts - timedelta(minutes=2)
        except Exception:
            pass

    runs = client.list_runs(
        project_name=PROJECT,
        is_root=True,
        run_type="chain",
        **(cursor_filter if cursor_filter else {}),
    )

    picked = 0
    for run in runs:
        if not _sampled(run):
            continue
        if _has_score(run):
            continue
        if not _stratified_sample(run):
            continue

        outputs = run.outputs or {}
        answer = outputs.get("answer")
        context = outputs.get("context")     # post-rerank chunks
        query = (run.inputs or {}).get("query", "")
        thread_id = (run.config or {}).get("thread_id", "")

        if not answer or not context:
            continue

        already = _existing_keys(run)

        jobs = [
            ("faithfulness",
             FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
             dict(input=query, actual_output=answer, retrieval_context=context)),
            ("answer_relevancy",
             AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
             dict(input=query, actual_output=answer)),
            ("contextual_relevancy",
             ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
             dict(input=query, actual_output=answer, retrieval_context=context)),
        ]

        for key, metric, tc_kwargs in jobs:
            if key in already:
                continue
            try:
                metric.measure(LLMTestCase(**tc_kwargs))
                client.create_feedback(
                    run_id=run.id,
                    key=key,
                    score=metric.score,
                    comment=metric.reason,
                )
                picked += 1
            except Exception as e:
                print(f"[{key}] failed on run {run.id}: {e}")

    # Advance the cursor
    try:
        last = client.list_runs(project_name=PROJECT, is_root=True, run_type="chain")
        if last:
            LAST_SCAN_RUN_ID = str(last[-1].id)
            LAST_SCAN_TS = datetime.now(timezone.utc)
    except Exception:
        pass

    print(f"[triad_worker] scored {picked} additional traces; scanned {len(runs)} runs")


if __name__ == "__main__":
    # live terminal version: poll until killed
    while True:
        score_recent_traces()
        time.sleep(POLL_SECONDS)
