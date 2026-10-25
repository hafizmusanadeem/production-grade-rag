"""
Hermetic test setup: app.main validates env vars and builds LLM clients at import
time, so give the suite dummy values (setdefault keeps any value you export
yourself) and make sure no test ever ships telemetry to Logfire.
"""
import os

for _name in (
    "GEMINI_API_KEY",
    "QDRANT_API_KEY",
    "QDRANT_CLUSTER_ENDPOINT",
    "QDRANT_COLLECTION",
    "LOGFIRE_TOKEN",
    "PORTKEY_API_KEY",
    "PORTKEY_SLUG_1",
    "PORTKEY_SLUG_2",
    "PORTKEY_CONFIG_ID",
):
    os.environ.setdefault(_name, "test-dummy")

os.environ["QDRANT_CLUSTER_ENDPOINT"] = os.environ.get(
    "QDRANT_CLUSTER_ENDPOINT", "https://example.invalid"
)

import logfire  # noqa: E402

_original_configure = logfire.configure


def _configure_without_sending(*args, **kwargs):
    kwargs["send_to_logfire"] = False
    return _original_configure(*args, **kwargs)


logfire.configure = _configure_without_sending


def pytest_collection_modifyitems(config, items):
    import pytest
    for item in items:
        if item.get_closest_marker("eval") is None:
            item.add_marker(pytest.mark.unit)
