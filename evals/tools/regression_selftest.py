"""Prove the regression gate actually catches a broken retriever.

Runs the suite in a subprocess with a deliberately broken RERANK_MIN_SCORE, then
compares against a baseline. The selftest PASSES only when compare exits 1 (FAIL)
and names agent.answer_rate -- i.e. the gate really fires, not just the harness.
"""
import argparse
import os
import subprocess
import sys

BREAK_OUT = "baselines/candidate_break.json"
BREAK_ENV = {"RERANK_MIN_SCORE": "0.9"}


def _default_runner(cmd, env=None):
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def evaluate(baseline, runner=_default_runner, *, limit=20, quiet=True):
    """Return (ok, detail) describing whether the gate caught the break."""
    suite_cmd = [sys.executable, "-m", "evals.run_suite",
                 "--limit", str(limit), "--out", BREAK_OUT]
    if quiet:
        suite_cmd.append("--quiet")
    env = dict(os.environ)
    env.update(BREAK_ENV)
    suite_code, suite_out = runner(suite_cmd, env)
    if suite_code not in (0, 1):
        return False, f"suite run itself crashed (exit {suite_code}); gate untested"

    compare_cmd = [sys.executable, "-m", "evals.compare",
                   "--baseline", baseline, "--candidate", BREAK_OUT]
    code, out = runner(compare_cmd)
    mentioned = "agent.answer_rate" in out
    if code == 1 and mentioned:
        return True, "gate fired: compare exited 1 and flagged agent.answer_rate"
    return False, (f"expected exit 1 mentioning agent.answer_rate, got exit {code} "
                   f"(mentioned={mentioned})")


def main():
    ap = argparse.ArgumentParser(description="Self-test the regression gate.")
    ap.add_argument("--baseline", required=True,
                    help="path to the baseline snapshot to compare against")
    args = ap.parse_args()
    ok, detail = evaluate(args.baseline)
    print(f"SELFTEST {'PASS' if ok else 'FAIL'}: {detail}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
