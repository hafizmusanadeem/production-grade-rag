"""Draft routing / unanswerable / multiturn golden sets.

Deterministically derived from retriever.json where an item has an answerable
Kubernetes question. Every item carries "status": "draft" -- a human must
review and remove the status before any baseline is built on it.
"""
import json
import random
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "goldens"
OUT_IDS = set()

CONVERSATIONAL = [
    "hello",
    "hi there",
    "thanks, that helped",
    "thank you",
    "what did I just ask you?",
    "can you repeat your last answer?",
    "and in plain English?",
    "what do you mean by that?",
    "ok, now explain it simply",
    "can you give me an example?",
    "you are great, thanks",
    "good bye",
]

OUT_OF_SCOPE = [
    "What is the capital of Germany?",
    "Plan a 5-day trip to Bali.",
    "Write a Django shop checkout view.",
    "Write a poem about spring flowers.",
    "Who won the 2022 FIFA World Cup?",
    "Give me a lasagna recipe.",
    "Explain quantum computing to a child.",
    "Invest my savings in penny stocks.",
    "Translate 'good morning' into Japanese.",
    "Yar laptop buy karne ke liye kya recommend karte ho?",  # Roman Urdu
    "Mausam ka haal kya hai Kal sta?",                     # Roman Urdu / Hinglish
    "Aaj cricket ka score kya hai?",
    "Write my resume objective statement.",
    "What is a good Netflix series for tonight?",
]

MIXED = [
    "Explain Kubernetes CronJobs, then recommend a laptop for a student.",
]

UNANSWERABLE = [
    "How do I upgrade a kubeadm-provisioned control plane from 1.30 to 1.31?",
    "How is mutual TLS (mTLS) enforced between pods by Istio?",
    "What are the internals of the CSI driver attach/detach sequence?",
    "How do I take an etcd snapshot and restore it on a new control plane?",
    "How do I install cert-manager with Jetstack Helm repository?",
    "How does OPA Gatekeeper validate admission policies against a ConstraintTemplate?",
    "How do I set up the Kubernetes dashboard behind an Ingress?",
    "What does a Calico CNI BGP configuration look like?",
    "How do I build a Cluster API management cluster for AWS?",
    "How is the Kubernetes node problem detector configured?",
    "How do I enable the Windows HostProcess containers feature gate?",
    "What is the reconciliation loop of the cloud-controller-manager?",
    "How do I configure a Kustomize overlay for multiple environments?",
    "How is the Vertical Pod Autoscaler admission controller installed?",
    "How do I deploy the NVIDIA GPU Operator with MIG support?",
    "What is the on-disk format of an etcd WAL segment?",
    "How do I use kubectl diff server-side apply for a three-way merge?",
    "How is dynamic Kubernetes auditing configured to stream to a webhook?",
    "How do I set up Sigma-based rule evaluation in Falco for pods?",
    "How do I bootstrap a Kubernetes scheduler using a custom scheduling framework?",
]

MULTITURN = [
    # (turns, expected_intent, expected_behavior)
    ([{"role": "user", "content": "What is a Kubernetes CronJob?"},
      {"role": "user", "content": "What did I just ask you?"}],
     "conversational", "restates the CronJob question"),
    ([{"role": "user", "content": "Which restart policies can a Job Pod use?"},
      {"role": "user", "content": "Which one of those is not allowed?"}],
     "conversational", "refers to the restart policies named in the previous turn"),
    ([{"role": "user", "content": "How is the database-backup CronJob configured?"},
      {"role": "user", "content": "And what schedule does it use?"}],
     "conversational", "answers the follow-up from the previous turn's content"),
    ([{"role": "user", "content": "What does the ReplicaSet Controller do?"},
      {"role": "assistant", "content": "It reconciles pod counts."},
      {"role": "user", "content": "What happens after it is created?"}],
     "conversational", "continues the thread from the assistant's previous answer"),
    ([{"role": "user", "content": "How do I set parallelism in a work queue job?"},
      {"role": "assistant", "content": "Use the parallelism field in job.yaml."},
      {"role": "user", "content": "What is the recommended value?"}],
     "conversational", "answers the follow-up from the previous turn"),
    ([{"role": "user", "content": "Hello!"},
      {"role": "assistant", "content": "Hello! Ask me about Kubernetes."},
      {"role": "user", "content": "Great, thanks."}],
     "conversational", "closes the conversation politely"),
    ([{"role": "user", "content": "What is the difference between Indexed and NonIndexed modes?"},
      {"role": "assistant", "content": "Indexed gives each pod an ordinal."},
      {"role": "user", "content": "Can you say that again more simply?"}],
     "conversational", "re-explains from history"),
    ([{"role": "user", "content": "How do I exclude a container from VPA scaling?"},
      {"role": "assistant", "content": "Use the resourcePolicy on the VPA."},
      {"role": "user", "content": "What is that field called?"}],
     "conversational", "restates the field named in the previous turn"),
    ([{"role": "user", "content": "What is a Kubernetes Job design mistake?"},
      {"role": "assistant", "content": "Using restartPolicy: Always."},
      {"role": "user", "content": "Why is that a mistake?"}],
     "conversational", "elaborates from the previous turn"),
    ([{"role": "user", "content": "What is a sidecar container?"},
      {"role": "assistant", "content": "It is a helper container in the same pod."},
      {"role": "user", "content": "What did I ask you first?"}],
     "conversational", "restates the very first question in the session"),
]


def write(name, items):
    path = GOLDEN_DIR / f"{name}.json"
    seen = set()
    for it in items:
        assert it["id"] not in seen, f"duplicate id {it['id']}"
        seen.add(it["id"])
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path.name} ({len(items)} items)")


def main():
    retriever = json.loads((GOLDEN_DIR / "retriever.json").read_text(encoding="utf-8"))

    # technical routing items: stratified across source files, deterministic.
    rng = random.Random(7)
    by_source = {}
    for it in retriever:
        by_source.setdefault(it["source"], []).append(it)
    technical = []
    per_source = 3
    for source in sorted(by_source):
        pool = list(by_source[source])
        rng.shuffle(pool)
        for it in pool[:per_source]:
            technical.append(it)

    routing = []
    n = 0
    for it in technical:
        n += 1
        routing.append({"id": f"r{n:03d}", "history": [], "message": it["query"],
                        "expected_intent": "technical", "derived_from": it["id"],
                        "tags": ["doc:" + it["source"].split(".")[0], "type:factual", "lang:en"],
                        "status": "draft"})
    for msg in CONVERSATIONAL:
        n += 1
        routing.append({"id": f"r{n:03d}", "history": [], "message": msg,
                        "expected_intent": "conversational",
                        "tags": ["type:conversational", "lang:en" if msg.isascii() else "lang:roman_urdu"],
                        "status": "draft"})
    for msg in OUT_OF_SCOPE:
        n += 1
        routing.append({"id": f"r{n:03d}", "history": [], "message": msg,
                        "expected_intent": "out_of_scope",
                        "tags": ["type:off_topic",
                                 "lang:roman_urdu" if not msg.isascii() else "lang:en"],
                        "status": "draft"})
    for msg in MIXED:
        n += 1
        routing.append({"id": f"r{n:03d}", "history": [], "message": msg,
                        "expected_intent": "technical",
                        "tags": ["type:mixed", "lang:en"], "status": "draft"})
    write("routing", routing)

    unanswerable = []
    for i, q in enumerate(UNANSWERABLE, 1):
        unanswerable.append({"id": f"u{i:03d}", "question": q,
                             "expected": "DECLINE_NO_CONTEXT", "tags": ["lang:en"],
                             "status": "draft"})
    write("unanswerable", unanswerable)

    multiturn = []
    for i, (turns, intent, behaviour) in enumerate(MULTITURN, 1):
        multiturn.append({"id": f"m{i:03d}", "turns": turns, "expected_intent": intent,
                          "expected_behavior": behaviour,
                          "tags": ["type:multiturn", "lang:en"], "status": "draft"})
    write("multiturn", multiturn)


if __name__ == "__main__":
    main()
