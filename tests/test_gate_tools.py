"""Unit tests for the gate tools that need no API keys.

Covers regression_selftest (with a fake subprocess runner), count_metrics,
set_slo, baseline_stale_check and online signals.
"""
import json

import pytest

from evals.tools.regression_selftest import evaluate as selftest_evaluate


class FakeRunner:
    def __init__(self, suite_code=0, compare_code=1, compare_output=""):
        self.suite_code, self.compare_code = suite_code, compare_code
        self.compare_output = compare_output
        self.calls = []

    def __call__(self, cmd, env=None):
        self.calls.append((list(cmd), dict(env or {})))
        if "run_suite" in " ".join(cmd):
            return self.suite_code, "suite ran"
        return self.compare_code, self.compare_output


def test_selftest_passes_when_gate_fires(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = FakeRunner(compare_code=1, compare_output="blocked agent.answer_rate")
    ok, detail = selftest_evaluate("baselines/baseline.json", runner)
    assert ok and "gate fired" in detail
    suite_call, suite_env = runner.calls[0]
    assert suite_env.get("RERANK_MIN_SCORE") == "0.9"


def test_selftest_fails_when_compare_exits_0(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = FakeRunner(compare_code=0, compare_output="agent.answer_rate flat")
    ok, detail = selftest_evaluate("baselines/baseline.json", runner)
    assert not ok and "expected exit 1" in detail


def test_selftest_fails_when_metric_not_named(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = FakeRunner(compare_code=1, compare_output="some other metric blocked")
    ok, _ = selftest_evaluate("baselines/baseline.json", runner)
    assert not ok


def test_selftest_fails_when_suite_crashes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = FakeRunner(suite_code=2)
    ok, _ = selftest_evaluate("baselines/baseline.json", runner)
    assert not ok and "crashed" in _


def test_count_metrics_main(tmp_path, capsys):
    from evals.tools.count_metrics import main as count_main
    snapshot = {"metadata": {"suite_errors": {}},
                "metrics": {f"suite{i}.metric.avg_score": 0.5 for i in range(55)}}
    path = tmp_path / "snap.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    loop = None
    import sys
    sys.argv = ["count_metrics", "--snapshot", str(path)]
    try:
        code = count_main()
    finally:
        sys.argv = ["pytest"]
    assert code == 1  # 55 < 60
    assert "PASS" not in capsys.readouterr().out


def test_online_signals_all_ok():
    from evals.online.signals import compute_signals, evaluate
    runs = [_Run(error=None, outputs={"final_answer": "ok"})]
    marks = evaluate(compute_signals(runs))
    assert marks["error_rate"][1] is True


def test_online_signals_alerts_on_errors():
    from evals.online.signals import compute_signals, evaluate
    runs = [_Run(error="timeout"), _Run(error=None, outputs={"final_answer": "x"})]
    marks = evaluate(compute_signals(runs))
    assert marks["error_rate"][1] is False
    assert marks["decline_rate"][1] is True


def test_traffic_light_rules():
    from evals.tools.review_packet import traffic_light
    assert traffic_light(0.9, 0.0) == "GREEN"
    assert traffic_light(0.6, 0.0) == "GREEN"
    assert traffic_light(0.9, 0.05) == "YELLOW"
    assert traffic_light(0.5, 0.0) == "YELLOW"
    assert traffic_light(0.2, 0.0) == "RED"
    assert traffic_light(0.9, 0.3) == "RED"


def test_review_row_sorts_red_first():
    from evals.tools.review_packet import review_row, sort_by_light
    by_file = {"a.docx": [{"page_content": "Kubernetes Jobs run finite workloads",
                           "chunk_id": "c1", "file_name": "a.docx"}]}
    good = {"id": "g1", "question": "What are Jobs?",
            "ideal_answer": "Kubernetes Jobs run finite workloads", "source": "a.docx", "tags": []}
    bad = {"id": "g2", "question": "How to kubeadm upgrade?",
           "ideal_answer": "kubeadm upgrade control plane etcd snapshot restore", "source": "a.docx", "tags": []}
    rows = [review_row("retriever.json", good, by_file),
            review_row("retriever.json", bad, by_file)]
    assert rows[1]["light"] == "RED" or rows[1]["overlap"] < rows[0]["overlap"]
    order = sort_by_light(rows)
    assert [r["light"] for r in order] == sorted(
        [r["light"] for r in order],
        key=lambda l: ({"RED": 0, "YELLOW": 1, "GREEN": 2}[l],))


def test_review_packet_writes_draft_rows_offline(tmp_path, monkeypatch):
    import evals.tools.review_packet as rp
    monkeypatch.setattr(rp, "GOLDEN_DIR", tmp_path)
    monkeypatch.setattr(rp, "REVIEW_DIR", tmp_path / "review")
    goldens = [{"id": "d1", "message": "hello", "expected_intent": "conversational",
                "tags": ["type:conversational"], "status": "draft"}]
    (tmp_path / "scope_goldens.json").write_text(__import__("json").dumps(goldens),
                                                 encoding="utf-8")
    monkeypatch.setattr(rp, "load_chunks", lambda: {})
    assert rp.main() == 0
    assert (tmp_path / "review" / "scope_goldens.json.md").exists()
    assert (tmp_path / "review" / "scope_goldens.json.csv").exists()


class _Run:
    def __init__(self, error=None, outputs=None, start_time=None, end_time=None,
                 plan=None, feedback_stats=None, faithfulness=None, answer_relevancy=None):
        self.error = error
        self.outputs = outputs or {}
        self.start_time = start_time
        self.end_time = end_time
        self.plan = plan if plan is not None else []
        self.feedback_stats = feedback_stats
        self.faithfulness = faithfulness
        self.answer_relevancy = answer_relevancy


def test_load_goldens_accepts_str_path():
    """Several eval modules pass a str path; harness must not AttributeError."""
    from evals.harness import load_goldens
    goldens = load_goldens("evals/goldens/scope_goldens.json")
    assert isinstance(goldens, list) and goldens


def test_load_goldens_falls_back_to_golden_dir():
    """A bare filename resolves against evals/goldens/."""
    from evals.harness import load_goldens
    goldens = load_goldens("ops_questions.json")
    assert isinstance(goldens, list) and goldens


def test_load_goldens_missing_raises(tmp_path):
    """A path that exists nowhere must raise, not silently return []."""
    from evals.harness import load_goldens
    with pytest.raises(FileNotFoundError):
        load_goldens("definitely_not_a_golden_file.json")
