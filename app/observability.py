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
    _configured = True
