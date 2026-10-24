import os

os.environ["ENVIRONMENT"] = "eval"

import logfire  # noqa: E402

if not getattr(logfire.configure, "_eval_patched", False):
    _original_configure = logfire.configure

    def _configure_without_sending(*args, **kwargs):
        kwargs["send_to_logfire"] = False
        return _original_configure(*args, **kwargs)

    _configure_without_sending._eval_patched = True
    logfire.configure = _configure_without_sending
