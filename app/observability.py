import logfire
from app.config import settings

_configured = False


def configure_logfire() -> None:
    global _configured
    if _configured:
        return

    logfire.configure(
        token=settings.LOGFIRE_TOKEN,
        service_name=settings.LOGFIRE_SERVICE_NAME,
        environment=settings.ENVIRONMENT,
        send_to_logfire="if-token-present",
    )

    # Process-wide instrumentation: every outbound HTTP call becomes a span.
    logfire.instrument_requests()
    logfire.instrument_httpx()          # Qdrant, Portkey, OpenAI-compatible clients
    logfire.instrument_system_metrics() # CPU / memory (you already pin the OTel package)

    _configured = True

def instrument_fastapi(fastapi_app) -> None:
    logfire.instrument_fastapi(fastapi_app)
