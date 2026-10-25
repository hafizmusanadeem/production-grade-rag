"""Atomic manifest and corpus fingerprint."""
import hashlib
import json
import os
import tempfile
from pathlib import Path


def update_manifest(data_dir, entry):
    root=Path(data_dir); root.mkdir(parents=True,exist_ok=True); path=root/"manifest.json"
    try: current=json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except (OSError,json.JSONDecodeError): current=[]
    by_name={item.get("file_name"):item for item in current if item.get("file_name")}
    by_name[entry["file_name"]]=dict(entry)
    payload=json.dumps([by_name[k] for k in sorted(by_name)],indent=2,sort_keys=True)
    temp=None
    try:
        with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=root,prefix=".manifest.",suffix=".tmp",delete=False) as f:
            temp=Path(f.name); f.write(payload); f.flush(); os.fsync(f.fileno())
        temp.replace(path)
    finally:
        if temp and temp.exists(): temp.unlink(missing_ok=True)
    return path


def corpus_fingerprint(data_dir):
    root=Path(data_dir); path=root/"manifest.json"
    entries=json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    point_count="unavailable"
    try:
        from app.config import settings
        from app.services.retrieval.qdrant_service import get_qdrant_client
        point_count=get_qdrant_client(timeout=settings.QDRANT_SEARCH_TIMEOUT_SECONDS).count(
            collection_name=settings.QDRANT_COLLECTION,exact=True).count
    except Exception: pass
    raw=json.dumps({"entries":sorted(entries,key=lambda x:x.get("file_name","")),"point_count":point_count},
                   sort_keys=True,separators=(",",":"))
    return hashlib.sha256(raw.encode()).hexdigest()
