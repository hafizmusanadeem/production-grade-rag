"""Recommend SLO_E2E_P95_MS from a measured snapshot. Prints only; never edits config."""
import argparse
import json
import math

from evals.config import CANDIDATE_PATH, SLO_E2E_P95_MS

MARGIN = 1.25


def recommend_slo(measured_ms: float) -> int:
    """Measured p95 x margin, rounded UP to the nearest 100 ms."""
    return int(math.ceil(measured_ms * MARGIN / 100.0) * 100)


def main():
    ap = argparse.ArgumentParser(description="Recommend the e2e p95 latency SLO.")
    ap.add_argument("--snapshot", default=str(CANDIDATE_PATH))
    args = ap.parse_args()
    with open(args.snapshot, encoding="utf-8") as f:
        snapshot = json.load(f)
    measured = snapshot.get("metrics", {}).get("ops.latency.e2e_p95_ms")
    print(f"measured ops.latency.e2e_p95_ms = {measured} ms")
    if measured is None or (isinstance(measured, float) and math.isnan(measured)):
        print("No measured p95 in this snapshot (ops suite may have failed). "
              "Re-run evals.eval_files.eval_ops before using this number.")
        return 1
    print(f"current  SLO_E2E_P95_MS      = {SLO_E2E_P95_MS:.0f} ms")
    print(f"suggested SLO_E2E_P95_MS     = {recommend_slo(measured)} ms "
          f"(measured x {MARGIN}, rounded up to 100 ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
