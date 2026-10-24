"""evals/config.py
Paths, judge model, thresholds, and env overrides for the eval suite.
"""
import os
from pathlib import Path

# Root of the project (this package's parent is the repo root)
ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT / "evals" / "goldens"
ARTIFACTS_DIR = ROOT / "evals" / "artifacts"

EVALS_DIR = ROOT   # this package lives at repo root's evals/
HELPER_DIR = ROOT / "evals" / "helper"
TOOLS_DIR = ROOT / "evals" / "tools"
ONLINE_DIR = ROOT / "evals" / "online"
BASELINE_PATH = ROOT / "baselines" / "baseline.json"
CANDIDATE_PATH = ROOT / "baselines" / "candidate.json"

DEFAULT_THRESHOLD = 0.7

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "openai/gpt-oss-20b:free")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
RETRIEVAL_LIMIT_K = int(os.getenv("RETRIEVAL_LIMIT_K", "3"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.02"))

# Portkey responder config id (fallback config)
RESPONDER_CONFIG_ID = os.getenv("PORTKEY_CONFIG_ID", "fallbacks")

# Operational SLOs
SLO_E2E_P95_MS = float(os.getenv("SLO_E2E_P95_MS", "3000"))
SLO_TTFT_P95_MS = float(os.getenv("SLO_TTFT_P95_MS", "1200"))
