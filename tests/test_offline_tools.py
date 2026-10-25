"""Unit tests for the offline gate tools (no API keys needed)."""
import json

from evals.run_suite import resolved_models
from evals.target import EvalResult
from evals.tools.baseline_stale_check import drift
from evals.tools.count_metrics import summarise
from evals.tools.set_slo import recommend_slo


def _result(**usage):
    return EvalResult(query="q", answer="a", context=[], usage=usage)


def test_resolved_models_single_model():
    out = resolved_models([_result(planner_model="a", responder_model="b"),
                           _result(planner_model="a", responder_model="b")])
    assert out == {"resolved_planner_models": ["a"], "resolved_responder_models": ["b"]}


def test_resolved_models_fallback_unknown():
    out = resolved_models([_result(planner_model="a"), _result()])
    assert out["resolved_planner_models"] == ["a", "unknown"]


def test_recommend_slo_rounding():
    assert recommend_slo(4000.0) == 5000
    assert recommend_slo(2900.0) == 3700
    assert recommend_slo(299.0) == 400


def test_summarise_counts_and_target():
    snapshot = {"metadata": {"suite_errors": {}},
                "metrics": {"retriever.source_hit_at_k": 0.9, "ops.cost.cost_per_query_usd": 0.01}}
    out = summarise(snapshot, target=1)
    assert out["non_info"] == 2
    assert out["meets_target"] is True
    out2 = summarise({"metadata": {}, "metrics": {}}, target=60)
    assert out2["meets_target"] is False


def test_drift_detects_settings_change(monkeypatch):
    md = {"prompt_hash": "nope",
          "settings": {"chunk_size": 999, "chunk_overlap": 200,
                       "embedding_model": "gemini-embedding-001", "retrieval_limit_k": 10,
                       "rerank_top_n": 5, "rerank_min_score": 0.02},
          "golden_hashes": {}, "judge_model": "x"}
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "x")
    rows = drift(md)
    fields = [r[0] for r in rows]
    assert "settings.chunk_size" in fields


def test_check_golden_status(tmp_path):
    from evals.tools.check_golden_status import render, scan
    good = tmp_path / "g.json"
    good.write_text(json.dumps([{"id": "a"}, {"id": "b", "status": "draft"}]), encoding="utf-8")
    rows = scan(tmp_path)
    assert rows[0]["draft"] == 1 and rows[0]["total"] == 2
    assert "draft" in render(rows)
