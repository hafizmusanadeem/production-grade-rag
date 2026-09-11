"""
langchain_core.language_models.base does, at MODULE level (not lazily):

    try:
        from transformers import GPT2TokenizerFast
        _HAS_TRANSFORMERS = True
    except ImportError:
        _HAS_TRANSFORMERS = False

This is a fallback tokenizer used only when tiktoken doesn't recognize the
model. We use Azure/OpenAI models, which tiktoken already handles - so this
fallback is realistically dead code for us, but its import still costs
several seconds at every startup because `transformers` (needed elsewhere,
e.g. by flashrank) is installed in this venv.

Fix: insert a lightweight fake `transformers` module into sys.modules
*before* langchain_core's import chain runs. It has no `GPT2TokenizerFast`
attribute, so `from transformers import GPT2TokenizerFast` raises
ImportError immediately (caught by langchain_core's own except clause) -
without ever executing the real, heavy transformers/__init__.py.

We remove the stub again right after, so anything that imports the REAL
transformers later (e.g. flashrank's lazy `Ranker()` on first rerank call)
gets the genuine package, unaffected.

Tradeoff: langchain_core's own GPT2TokenizerFast-based token-count fallback
is permanently unavailable for the life of this process (cached in
_HAS_TRANSFORMERS). Acceptable here since tiktoken covers our models.

Import this module FIRST, before any langchain/app.agents/app.gateway
imports, e.g. at the very top of app/main.py:

    import app.startup_patches  # noqa: F401  (must be first import)
"""
import sys
import types

_stub_inserted = False

if "transformers" not in sys.modules:
    _stub = types.ModuleType("transformers")
    sys.modules["transformers"] = _stub
    _stub_inserted = True


def restore_real_transformers() -> None:
    """Call this once the langchain import chain has finished, so later
    genuine uses of transformers (e.g. flashrank) get the real package."""
    global _stub_inserted
    if _stub_inserted and isinstance(sys.modules.get("transformers"), types.ModuleType):
        # Only remove it if it's still our stub (no __file__/real attrs)
        if not hasattr(sys.modules["transformers"], "__file__"):
            del sys.modules["transformers"]
        _stub_inserted = False
