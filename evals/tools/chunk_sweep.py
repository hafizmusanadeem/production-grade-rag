"""Offline chunking sweep. No API keys, no Qdrant.
python -m evals.tools.chunk_sweep --docs DATA/true_data --show 8
"""
import argparse, json, random, statistics
from pathlib import Path
from app.config import settings
from app.ingestion.chunking.chunker import chunk_documents
from app.ingestion.loaders import load_document
from evals.tools.review_packet import tokens

CONFIGS = [(500, 50), (1000, 200), (1500, 300), (2000, 400), (100,10), (200, 20), (300,30), (400,40), (600,60), (700, 70), (800, 80), (900, 90), (150, 25)]
GOLDEN = Path("evals/goldens/retriever.json")

def chunk_all(files, size, overlap):
    settings.CHUNK_SIZE, settings.CHUNK_OVERLAP = size, overlap  # read at call time
    return {f.name: [c.page_content for c in chunk_documents(load_document(f))] for f in files}

def overlap(ideal, chunk):
    return len(ideal & tokens(chunk)) / len(ideal) if ideal else 0.0

def score(goldens, by_file):
    best1, best3 = [], []
    for g in goldens:
        ideal = tokens(g["ideal_answer"])
        ranked = sorted(by_file.get(g["source"], []), key=lambda c: overlap(ideal, c), reverse=True)
        if not ranked:
            best1.append(0.0); best3.append(0.0); continue
        best1.append(overlap(ideal, ranked[0]))
        union = set().union(*(tokens(c) for c in ranked[:3]))
        best3.append(len(ideal & union) / len(ideal) if ideal else 0.0)
    n = len(goldens)
    return {"mean_best1": statistics.mean(best1),
            "pct_best1>=0.6": sum(x >= 0.6 for x in best1) / n,
            "mean_top3_union": statistics.mean(best3),
            "pct_top3>=0.8": sum(x >= 0.8 for x in best3) / n}

def shape(by_file):
    lens = [len(c) for cs in by_file.values() for c in cs]
    clean_end = sum(c.rstrip()[-1:] in ".:!?)}]`" for cs in by_file.values() for c in cs)
    return {"chunks": len(lens), "p50_len": statistics.median(lens),
            "min_len": min(lens), "pct_tiny(<150)": sum(l < 150 for l in lens) / len(lens),
            "pct_clean_end": clean_end / len(lens)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--show", type=int, default=0, help="print N random chunks per config")
    args = ap.parse_args()
    files = [p for p in Path(args.docs).rglob("*") if p.suffix.lower() in
             {".pdf", ".docx", ".pptx", ".html", ".htm", ".txt", ".md"}]
    goldens = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for size, ov in CONFIGS:
        by_file = chunk_all(files, size, ov)
        print(f"\n=== size={size} overlap={ov} ===")
        print(" shape :", {k: round(v, 3) for k, v in shape(by_file).items()})
        print(" cover :", {k: round(v, 3) for k, v in score(goldens, by_file).items()})
        if args.show:
            pool = [c for cs in by_file.values() for c in cs]
            for c in random.Random(0).sample(pool, min(args.show, len(pool))):
                print("-" * 60); print(c)

if __name__ == "__main__":
    main()
