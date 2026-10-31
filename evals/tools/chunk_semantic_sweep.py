"""Semantic chunk size/overlap sweep with a local vector DB. Free, offline, no API keys.

Stack: fastembed (local ONNX embeddings) + Qdrant local mode (on-disk, no server)
       + your own loaders and the same RecursiveCharacterTextSplitter as production.

For every (chunk_size, overlap) config:
  1. load + chunk the docs, embed the chunks (cached across configs by text hash)
  2. store them in their own local Qdrant collection
  3. run every retriever.json golden query, take top-k
  4. score semantically against the golden's ideal_answer

Metrics (all 0-1, higher is better):
  source_hit   - any top-k chunk comes from the golden's source file
  answer_sim   - max cosine(ideal_answer, retrieved chunk) in top-k   (semantic coverage)
  token_recall - share of the ideal answer's key tokens present in the top-k text
  precision    - share of top-k chunks that come from the right source file
  score        - mean(source_hit, answer_sim, token_recall)  -> ranking key

Setup:
    pip install fastembed qdrant-client langchain-text-splitters numpy
    or
    uv add fastembed qdrant-client langchain-text-splitters numpy
Run from project root:
    python -m evals.tools.chunk_semantic_sweep
    python -m evals.tools.chunk_semantic_sweep --docs DATA/true_data --k 5 --limit 40
    python -m evals.tools.chunk_semantic_sweep --configs 500:50,1000:200,1500:300
"""
import argparse
import hashlib
import json
import re
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.ingestion.loaders import load_document

SUPPORTED = {".pdf", ".docx", ".pptx", ".html", ".htm", ".txt", ".md"}
GOLDEN = Path("evals/goldens/retriever.json")
DB_DIR = Path("DATA_eval/chunk_sweep_qdrant")          # gitignored
OUT_JSON = Path("DATA_eval/chunk_sweep_results.json")
DEFAULT_CONFIGS = [(300, 30), (500, 50), (500, 100), (800, 80), (1000, 100),
                   (1000, 200), (1500, 150), (1500, 300), (2000, 200), (2000, 400)]
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]               # same as production chunker


def key_tokens(text):
    return {t.lower() for t in re.findall(r'\b[A-Za-z][A-Za-z0-9_.-]{2,}\b|\b\d+(?:\.\d+)*\b', str(text))}


class Embedder:
    """fastembed wrapper with a text-hash cache so identical chunks embed once."""

    def __init__(self, model_name):
        self.model = TextEmbedding(model_name=model_name)
        self.cache = {}

    def passages(self, texts):
        keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        missing = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in self.cache]
        if missing:
            for (k, _), vec in zip(missing, self.model.passage_embed([t for _, t in missing])):
                self.cache[k] = np.asarray(vec, dtype=np.float32)
        return np.stack([self.cache[k] for k in keys])

    def queries(self, texts):
        return np.stack([np.asarray(v, dtype=np.float32) for v in self.model.query_embed(texts)])


def load_docs(folder):
    files = sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not files:
        raise SystemExit(f"No supported files under {folder}")
    docs = {}
    for f in files:
        try:
            docs[f.name] = load_document(f)
        except Exception as exc:
            print(f"  skip {f.name}: {exc}")
    return docs


def chunk_all(docs, size, overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,
                                              length_function=len, separators=SEPARATORS)
    chunks = []
    for name, loaded in docs.items():
        for di, doc in enumerate(loaded):
            for ci, text in enumerate(splitter.split_text(doc.page_content)):
                if text.strip():
                    chunks.append({"file_name": name, "doc_index": di, "chunk_index": ci, "text": text})
    return chunks


def build_collection(client, name, chunks, vectors):
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config=qm.VectorParams(size=vectors.shape[1],
                                                                  distance=qm.Distance.COSINE))
    batch = 128
    for s in range(0, len(chunks), batch):
        client.upsert(name, points=[
            qm.PointStruct(id=s + i, vector=vectors[s + i].tolist(),
                           payload={"file_name": c["file_name"], "text": c["text"]})
            for i, c in enumerate(chunks[s:s + batch])
        ], wait=True)


def evaluate_config(client, emb, docs, goldens, ideal_vecs, query_vecs, size, overlap, k):
    t0 = time.perf_counter()
    chunks = chunk_all(docs, size, overlap)
    if not chunks:
        return None
    vectors = emb.passages([c["text"] for c in chunks])
    name = f"sweep_{size}_{overlap}"
    build_collection(client, name, chunks, vectors)

    hits, sims, recalls, precs = [], [], [], []
    for g, ideal_vec, qvec in zip(goldens, ideal_vecs, query_vecs):
        res = client.query_points(name, query=qvec.tolist(), limit=k, with_payload=True).points
        texts = [p.payload["text"] for p in res]
        files = [p.payload["file_name"] for p in res]
        hits.append(float(g["source"] in files))
        precs.append(files.count(g["source"]) / len(files) if files else 0.0)
        if texts:
            tv = emb.passages(texts)
            denom = np.linalg.norm(tv, axis=1) * np.linalg.norm(ideal_vec) + 1e-9
            sims.append(float(np.max(tv @ ideal_vec / denom)))
        else:
            sims.append(0.0)
        ideal_tokens = key_tokens(g["ideal_answer"])
        got = key_tokens(" ".join(texts))
        recalls.append(len(ideal_tokens & got) / len(ideal_tokens) if ideal_tokens else 0.0)

    lens = [len(c["text"]) for c in chunks]
    m = statistics.mean
    row = {"size": size, "overlap": overlap, "chunks": len(chunks),
           "avg_len": round(m(lens)), "pct_tiny": round(sum(l < 150 for l in lens) / len(lens), 3),
           "source_hit": round(m(hits), 4), "answer_sim": round(m(sims), 4),
           "token_recall": round(m(recalls), 4), "precision": round(m(precs), 4),
           "seconds": round(time.perf_counter() - t0, 1)}
    row["score"] = round(m([row["source_hit"], row["answer_sim"], row["token_recall"]]), 4)
    return row


def parse_configs(text):
    out = []
    for part in text.split(","):
        size, overlap = (int(x) for x in part.split(":"))
        if overlap >= size:
            raise SystemExit(f"overlap must be < size: {part}")
        out.append((size, overlap))
    return out


def print_table(rows):
    cols = ["size", "overlap", "chunks", "avg_len", "pct_tiny", "source_hit",
            "answer_sim", "token_recall", "precision", "score"]
    print("\n" + " ".join(f"{c:>12}" for c in cols))
    print("-" * (13 * len(cols)))
    for r in sorted(rows, key=lambda r: r["score"], reverse=True):
        print(" ".join(f"{r[c]:>12}" for c in cols))


def main():
    ap = argparse.ArgumentParser(description="Semantic chunk size/overlap sweep (local, free).")
    ap.add_argument("--docs", default="DATA/true_data", help="folder of source documents")
    ap.add_argument("--golden", default=str(GOLDEN))
    ap.add_argument("--k", type=int, default=5, help="top-k chunks retrieved per query")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N goldens")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="fastembed model name")
    ap.add_argument("--configs", default=None, help="e.g. 500:50,1000:200")
    ap.add_argument("--keep-db", action="store_true", help="don't wipe the local DB first")
    args = ap.parse_args()

    configs = parse_configs(args.configs) if args.configs else DEFAULT_CONFIGS
    goldens = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    if args.limit:
        goldens = goldens[:args.limit]

    if DB_DIR.exists() and not args.keep_db:
        shutil.rmtree(DB_DIR)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(DB_DIR))

    print(f"Loading docs from {args.docs} ...")
    docs = load_docs(args.docs)
    present = set(docs)
    missing = {g["source"] for g in goldens} - present
    if missing:
        print(f"WARNING: goldens reference files not in --docs (will always miss): {sorted(missing)}")

    print(f"Loading embedding model {args.model} (first run downloads ~130MB) ...")
    emb = Embedder(args.model)
    query_vecs = emb.queries([g["query"] for g in goldens])
    ideal_vecs = emb.passages([g["ideal_answer"] for g in goldens])

    rows = []
    for size, overlap in configs:
        print(f"\n=== size={size} overlap={overlap} ===")
        row = evaluate_config(client, emb, docs, goldens, ideal_vecs, query_vecs, size, overlap, args.k)
        if row:
            rows.append(row)
            print(f"  chunks={row['chunks']} score={row['score']} "
                  f"hit={row['source_hit']} sim={row['answer_sim']} recall={row['token_recall']} "
                  f"({row['seconds']}s)")

    print_table(rows)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"k": args.k, "model": args.model, "n_goldens": len(goldens),
                                    "results": rows}, indent=2), encoding="utf-8")
    best = max(rows, key=lambda r: r["score"])
    print(f"\nBest: CHUNK_SIZE={best['size']} CHUNK_OVERLAP={best['overlap']} (score {best['score']})")
    print(f"Results: {OUT_JSON}   Local DB: {DB_DIR}")
    client.close()


if __name__ == "__main__":
    main()