"""Unit tests for compare.py's verdict and --markdown output format (C2)."""

from evals.compare import classify, compare, markdown_report


class FakeRegistry:
    """Stub the registry so the tests stay independent of metric names."""

    def __init__(self, rules):
        self.rules = rules

    def rule_for(self, metric_id):
        return self.rules[metric_id]


BASE_META = {"git_sha": "aaa", "prompt_hash": "p1"}
CAND_META = {"git_sha": "bbb", "prompt_hash": "p2"}


def test_classify_gate_drop_is_blocked():
    rule = {"direction": "higher", "kind": "gate", "tol": 0.02, "rel_tol": 0.0}
    row = classify("m.avg_score", 0.8, 0.5, rule)
    assert row["status"] == "blocked"


def test_classify_guardrail_drop_is_regressed():
    rule = {"direction": "higher", "kind": "guardrail", "tol": 0.05, "rel_tol": 0.0}
    row = classify("m.avg_score", 0.8, 0.6, rule)
    assert row["status"] == "regressed"


def test_classify_zero_baseline_cost_not_flagged():
    """Zero baseline + relative tolerance must not flag any nonzero cost."""
    rule = {"direction": "lower", "kind": "guardrail", "tol": 1e-6, "rel_tol": 0.15}
    assert classify("ops.cost.cost_per_query_usd", 0.0, 0.0000005, rule)["status"] == "flat"
    assert classify("ops.cost.cost_per_query_usd", 0.0, 0.01, rule)["status"] == "regressed"


def test_markdown_has_marker_and_table():
    baseline = {"metadata": BASE_META,
                "metrics": {"safety.leakage.protected_avg_score": 0.95}}
    candidate = {"metadata": CAND_META, "suite_errors": {},
                 "metrics": {"safety.leakage.protected_avg_score": 0.5}}
    rules = {"safety.leakage.protected_avg_score":
             {"direction": "higher", "kind": "gate", "tol": 0.02, "rel_tol": 0.0}}
    import evals.compare as cmp_mod
    original = cmp_mod.rule_for
    cmp_mod.rule_for = rules.__getitem__
    try:
        verdict, rows = compare(baseline, candidate)
        md = markdown_report(verdict, rows, BASE_META, CAND_META,
                             "baselines/baseline.json", "baselines/candidate.json")
    finally:
        cmp_mod.rule_for = original
    assert verdict == "FAIL"
    assert md.startswith("<!-- evals-compare -->")
    assert "| metric | baseline | candidate | delta | kind | status |" in md
    assert "`safety.leakage.protected_avg_score`" in md
    assert "Verdict: FAIL" in md
    assert "blocked" in md


def test_markdown_flat_run_is_calm():
    metrics = {"m.avg_score": 0.8}
    rules = {"m.avg_score": {"direction": "higher", "kind": "guardrail", "tol": 0.05, "rel_tol": 0.0}}
    import evals.compare as cmp_mod
    original = cmp_mod.rule_for
    cmp_mod.rule_for = rules.__getitem__
    try:
        verdict, rows = compare({"metadata": BASE_META, "metrics": metrics},
                                {"metadata": BASE_META, "metrics": metrics})
        md = markdown_report(verdict, rows, BASE_META, BASE_META, "b.json", "c.json")
    finally:
        cmp_mod.rule_for = original
    assert verdict == "PASS"
    assert "No gate or guardrail row changed" in md
