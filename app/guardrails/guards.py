import re
from dataclasses import dataclass, field

from app.agents.prompts import CANARY
from app.exceptions import InputRejectedError
from app.guardrails.rules import LEAK_REFUSAL_MESSAGE, SECRET_PATTERNS


@dataclass
class GuardResult:
    answer: str
    blocked: bool = False
    reasons: list[str] = field(default_factory=list)


def input_guard(query):
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", str(query or ""))
    value = re.sub(r"\\s+", " ", value).strip()
    if CANARY.casefold() in value.casefold():
        raise InputRejectedError("Internal marker included in input")
    return value


def output_guard(answer):
    value = str(answer or "")
    if CANARY.casefold() in value.casefold():
        return GuardResult(LEAK_REFUSAL_MESSAGE, True, ["canary_leak"])
    reasons = []
    for pattern in SECRET_PATTERNS:
        value, count = pattern.subn("[REDACTED]", value)
        if count:
            reasons.append("secret_redacted")
    return GuardResult(value, False, reasons)
