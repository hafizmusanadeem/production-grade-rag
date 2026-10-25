import pytest

from app.config import settings, validate_env_vars


def test_offline_eval_can_skip_logfire(monkeypatch):
    for name,value in {"LOGFIRE_TOKEN":None,"GEMINI_API_KEY":"x","QDRANT_API_KEY":"x",
        "QDRANT_CLUSTER_ENDPOINT":"https://example.invalid","QDRANT_COLLECTION":"test",
        "QDRANT_INGEST_TIMEOUT_SECONDS":120,"QDRANT_SEARCH_TIMEOUT_SECONDS":30,
        "QDRANT_UPSERT_BATCH_SIZE":3,"QDRANT_RETRY_BASE_DELAY_SECONDS":2.0,
        "QDRANT_MAX_RETRIES":5,"PORTKEY_API_KEY":"x","PORTKEY_SLUG_1":"x",
        "PORTKEY_SLUG_2":"x","PORTKEY_CONFIG_ID":"x"}.items():
        monkeypatch.setattr(settings,name,value)
    validate_env_vars(require_logfire=False)


def test_logfire_required_by_default(monkeypatch):
    monkeypatch.setattr(settings,"LOGFIRE_TOKEN",None)
    with pytest.raises(ValueError,match="LOGFIRE_TOKEN"): validate_env_vars()
