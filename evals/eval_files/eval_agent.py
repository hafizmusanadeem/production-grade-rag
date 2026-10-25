"""evals/eval_files/eval_agent.py -- agent-behaviour evals.

All metrics except the optional memory GEval are deterministic: they read the
planner's intent, whether the agent declined, and whether the answer is empty.

  routing_accuracy     planner-only accuracy over routing.json (+ per-class recall)
  answer_rate          answerable goldens that produced a non-declined answer
  abstention_accuracy  unanswerable goldens the agent declined on
  memory_accuracy      multiturn sessions whose final turn is conversational
"""
import argparse

from app.agents.nodes.decline import NO_CONTEXT_MESSAGE
from app.agents.nodes.planner import planner_node
from evals.config import GOLDEN_DIR
from evals.harness import load_goldens, select_goldens
from evals.target import AgentTarget

NAMESPACE = "agent"
ROUTING_PATH = GOLDEN_DIR / "routing.json"
UNANSWERABLE_PATH = GOLDEN_DIR / "unanswerable.json"
MULTITURN_PATH = GOLDEN_DIR / "multiturn.json"
ANSWERABLE_PATH = GOLDEN_DIR / "retriever.json"

INTENTS = ("technical", "conversational", "out_of_scope")


def _planner_intent(message, history=None):
    """Run the planner node in isolation and return its intent label."""
    messages = list(history or [])
    messages.append({"role": "user", "content": message})
    state = planner_node({"messages": messages, "current_query": "", "intent": "",
                          "documents": [], "plan": [], "status": "", "final_answer": ""})
    return state.get("intent", "")


def run_routing(*, limit=None, ids=None, verbose=True):
    goldens = select_goldens(load_goldens(ROUTING_PATH), limit=limit, ids=ids, seed=0)
    correct = 0
    per_class = {intent: {"hit": 0, "total": 0} for intent in INTENTS}
    for g in goldens:
        expected = g.get("expected_intent", "")
        observed = _planner_intent(g["message"], g.get("history") or [])
        bucket = per_class.setdefault(expected, {"hit": 0, "total": 0})
        bucket["total"] += 1
        if observed == expected:
            correct += 1
            bucket["hit"] += 1
        if verbose and observed != expected:
            print(f"  routing miss {g['id']}: expected={expected} got={observed}")
    n = len(goldens)
    out = {"routing_accuracy": correct / n if n else 0.0}
    for intent, bucket in per_class.items():
        out[f"{intent}_recall"] = bucket["hit"] / bucket["total"] if bucket["total"] else 0.0
    out["routing_n"] = n
    return out


def run_answer_rate(target: AgentTarget, *, limit=None, ids=None, verbose=True):
    goldens = select_goldens(load_goldens(ANSWERABLE_PATH), limit=limit, ids=ids,
                             seed=0, stratify_key="source")
    answered = 0
    for g in goldens:
        result = target.invoke(g["query"])
        if result.error or result.declined:
            continue
        if (result.answer or "").strip():
            answered += 1
    n = len(goldens)
    if verbose:
        print(f"  answer_rate: {answered}/{n}")
    return {"answer_rate": answered / n if n else 0.0, "answer_n": n}


def run_abstention(target: AgentTarget, *, limit=None, ids=None, verbose=True):
    goldens = select_goldens(load_goldens(UNANSWERABLE_PATH), limit=limit, ids=ids, seed=0)
    declined = no_context = 0
    for g in goldens:
        result = target.invoke(g.get("question") or g["message"])
        if result.declined:
            declined += 1
            if result.answer == NO_CONTEXT_MESSAGE:
                no_context += 1
        elif verbose:
            print(f"  abstention miss {g['id']}: answered an unanswerable question")
    n = len(goldens)
    return {"abstention_accuracy": declined / n if n else 0.0,
            "abstention_no_context_rate": no_context / n if n else 0.0, "abstention_n": n}


def run_memory(target: AgentTarget, *, limit=None, ids=None, verbose=True):
    goldens = select_goldens(load_goldens(MULTITURN_PATH), limit=limit, ids=ids, seed=0)
    correct = 0
    for g in goldens:
        turns = g.get("turns") or []
        if not turns:
            continue
        thread_id = f"eval-{g['id']}"
        result = None
        for turn in turns:
            if turn.get("role") != "user":
                continue
            result = target.invoke(turn["content"], thread_id=thread_id)
        if result is None:
            continue
        if result.intent == g.get("expected_intent"):
            correct += 1
        elif verbose:
            print(f"  memory miss {g['id']}: expected={g.get('expected_intent')} "
                  f"got={result.intent}")
    n = len(goldens)
    return {"memory_accuracy": correct / n if n else 0.0, "memory_n": n}


def run(target=None, *, limit=None, ids=None, verbose=True):
    target = target or AgentTarget()
    if verbose:
        print("[agent] routing (planner only, no retrieval)")
    out = run_routing(limit=limit, ids=ids, verbose=verbose)
    out.update(run_answer_rate(target, limit=limit, verbose=verbose))
    out.update(run_abstention(target, limit=limit, verbose=verbose))
    out.update(run_memory(target, limit=limit, verbose=verbose))
    return out


def main():
    ap = argparse.ArgumentParser(description="Agent-behaviour evals.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    ids = [x.strip() for x in args.ids.split(",") if x.strip()] if args.ids else None
    print(run(limit=args.limit, ids=ids, verbose=not args.quiet))


if __name__ == "__main__":
    main()
