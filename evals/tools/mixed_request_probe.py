"""Mixed-request probe (supports decision D-3).

For every PARTIAL item in scope_goldens.json and the mixed items in routing.json,
print the planner's intent and search query, the final answer, and whether the
answer drifts into the out-of-scope part (a keyword check via `forbidden_keywords`).

Emits one summary line: `mixed handled: X of Y`.

Needs live API keys. Run it; the human decides D-3 from the output.
"""
import argparse
import json

from evals.config import GOLDEN_DIR
from evals.target import AgentTarget

DEFAULT_FORBIDDEN = {
    "explain kubernetes cronjobs, then recommend a laptop for a student.": ["laptop", "recommend"],
}


def forbidden_for(golden):
    """Items may carry `forbidden_keywords`; otherwise fall back to a built-in map."""
    if golden.get("forbidden_keywords"):
        return list(golden["forbidden_keywords"])
    return DEFAULT_FORBIDDEN.get((golden.get("input") or golden.get("message") or "").strip().lower(), [])


def load_mixed_items():
    items = []
    for name, action_key in (("scope_goldens.json", "expected_action"), ("routing.json", "expected_intent")):
        path = GOLDEN_DIR / name
        if not path.exists():
            continue
        for item in json.loads(path.read_text(encoding="utf-8")):
            expected = str(item.get(action_key, "")).upper()
            if expected in ("PARTIAL", "MIXED") or "mixed" in (item.get("tags") or []):
                items.append((name, item))
    return items


def probe(golden, result):
    """ drifted=True when the answer mentions the out-of-scope part."""
    keywords = forbidden_for(golden)
    answer = (result.answer or "").lower()
    hits = [kw for kw in keywords if kw.lower() in answer]
    return hits, not hits


def render(name, golden, result, hits, handled):
    lines = [f"{'=' * 78}", f"{golden.get('id')}  ({name})", f"input      : {golden.get('input') or golden.get('message')}"]
    if golden.get("expected_action") or golden.get("expected_intent"):
        lines.append(f"expected   : {golden.get('expected_action') or golden.get('expected_intent')}")
    if keywords := forbidden_for(golden):
        lines.append(f"forbidden  : {', '.join(keywords)}")
    lines.append(f"intent     : {result.intent}")
    lines.append(f"declined   : {result.declined}")
    lines.append(f"plan       : {' | '.join(result.plan)}")
    lines.append(f"answer     : {(result.answer or '')[:600]}")
    lines.append(f"verdict    : {'handled (no drift)' if handled else f'DRIFTED on {hits}'}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Probe mixed-request handling.")
    ap.add_argument("--quiet", action="store_true", help="summary line only")
    args = ap.parse_args()

    items = load_mixed_items()
    target = AgentTarget()
    handled = total = 0
    for name, golden in items:
        result = target.invoke(golden.get("input") or golden.get("message"))
        hits, ok = probe(golden, result)
        total += 1
        handled += int(ok)
        if not args.quiet:
            print(render(name, golden, result, hits, ok))
    print(f"\nmixed handled: {handled} of {total}")
    print("Use this to decide D-3: keep the planner's PARTIAL behaviour, or change "
          "the goldens to expect DECLINE.")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
