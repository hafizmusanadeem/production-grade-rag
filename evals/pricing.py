"""Shadow pricing; no costs are emitted until official prices are verified."""
import os

PRICES = {}
DEFAULT_REFERENCE_MODEL = os.getenv("EVAL_REFERENCE_MODEL", "TODO_VERIFY_REFERENCE_MODEL")


def shadow_cost(prompt_tokens, completion_tokens, cached_tokens=0, model=DEFAULT_REFERENCE_MODEL):
    price = PRICES.get(model)
    if not price:
        return None
    prompt = max(int(prompt_tokens or 0), 0)
    cached = min(max(int(cached_tokens or 0), 0), prompt)
    completion = max(int(completion_tokens or 0), 0)
    return ((prompt-cached)*price["input_per_1m"] + cached*price.get("cached_per_1m", price["input_per_1m"])
            + completion*price["output_per_1m"]) / 1_000_000
