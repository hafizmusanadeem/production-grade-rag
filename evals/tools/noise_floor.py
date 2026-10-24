"""evals/tools/noise_floor.py -- two-snapshot delta report for tolerance calibration.

Runs two snapshots of the SAME code against the goldens + pipeline and prints
max/mean absolute delta per metric, suggesting a tolerance. The reference
tolerances (0.02 gates, 0.05 quality, 25% latency) were calibrated for a free
judge; recalibrate here if you pin a different judge (see D-1).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BASELINE = ROOT / "baselines" / "baseline.json"
CANDIDATE = ROOT / "baselines" / "candidate.json"


def main():
    if not BASELINE.exists() or not CANDIDATE.exists():
        print("Both baselines/*.json must exist.")
        raise SystemExit(1)

    b = json.loads(BASELINE.read_text())
    c = json.loads(CANDIDATE.read_text())
    bm = b.get("metrics", {})
    cm = c.get("metrics", {})

    all_keys = sorted(set(bm) | set(cm))
    deltas = []
    for k in all_keys:
        bv = bm.get(k, None)
        cv = cm.get(k, None)
        delta = None
        # only meaningful numeric deltas
        try:
            if isinstance(bv, float) and isinstance(cv, float):
                delta = cv - bv
        except Exception:
            pass
        deltas.append((k, bv, cv, delta))

    valid = [(k, bv, cv, d) for k, bv, cv, d in deltas if isinstance(d, (int, float))]
    if not valid:
        print("No numeric deltas to compare.")
        return

    max_delta = max(abs(d) for _, _, _, d in valid)
    avg_delta = sum(abs(d) for _, _, _, d in valid) / len(valid)
    suggested_tol = max(1.5 * max_delta, avg_delta)

    print("\nMetric noise floor analysis:")
    print(f"  samples     : {len(valid)}")
    print(f"  max abs delta : {max_delta:.4g}")
    print(f"  mean abs delta: {avg_delta:.4g}")
    print(f"  suggested tolerance : {suggested_tol:.4g}")

    # Split by kind
    kinds = {}
    for mid, bv, cv, d in valid:
        kinds.setdefault("baseline", []).append((mid, d))
        kinds["candidate"] = []

    # Just show top 20 biggest deltas per direction
    higher = sorted((m, d) for m, bv, cv, d in valid if isinstance(d, float) and d > 0)[-10:]
    lower = sorted((m, d) for m, bv, cv, d in valid if isinstance(d, float) and d < 0)[-10:]
    print(f"\nTop improvements ({len(higher)}):")
    for m, d in higher:
        print(f"  +{d:.4g}  {m}")
    print(f"\nTop regressions ({len(lower)}):")
    for m, d in lower:
        print(f"  {d:.4g}  {m}")


if __name__ == "__main__":
    main()
