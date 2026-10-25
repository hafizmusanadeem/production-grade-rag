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


# Portkey responder config id (fallback config)
RESPONDER_CONFIG_ID = os.getenv("PORTKEY_CONFIG_ID", "fallbacks")

# Operational SLOs
SLO_E2E_P95_MS = float(os.getenv("SLO_E2E_P95_MS", "3000"))
