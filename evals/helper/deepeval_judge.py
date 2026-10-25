import asyncio
import os
import random
import re
import time
from functools import lru_cache

from deepeval.models.base_model import DeepEvalBaseLLM
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from pydantic import BaseModel

load_dotenv(override=True)


def _text(msg) -> str:
    c = msg.content
    return c if isinstance(c, str) else "".join(
        p.get("text", "") if isinstance(p, dict) else str(p) for p in c
    )


class OpenRouterJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "openai/gpt-oss-20b:free"):
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY is required for the eval judge. Add it to .env")
        self.model_name = model_name
        timeout = float(os.getenv("EVAL_JUDGE_TIMEOUT", "60"))
        try:
            self.model = ChatOpenRouter(model=model_name, temperature=0, max_retries=2, timeout=timeout)
        except TypeError:
            self.model = ChatOpenRouter(model=model_name, temperature=0, max_retries=2)

    def load_model(self):
        return self.model

    @staticmethod
    def _json_hint(schema) -> str:
        return (
            "\n\nRespond with ONLY valid JSON that matches this JSON schema. "
            "No markdown fences, no extra text:\n"
            + str(schema.model_json_schema())
        )

    @staticmethod
    def _parse(text: str, schema):
        cleaned = re.sub(r"^```(?:json)?\\s*|\\s*```$", "", text.strip(), flags=re.I)
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Judge response did not contain a JSON object")
        return schema.model_validate_json(cleaned[start:end + 1])

    @staticmethod
    def _retryable(exc):
        return getattr(exc, "status_code", None) in {429, 500, 502, 503, 504} or any(
            text in str(exc).lower() for text in ("rate limit", "timeout", "temporarily unavailable"))

    def _invoke_with_backoff(self, prompt):
        for attempt in range(6):
            try:
                return self.model.invoke(prompt)
            except Exception as exc:
                if attempt == 5 or not self._retryable(exc):
                    raise
                time.sleep(min(60, 2 ** attempt + random.random()))

    async def _ainvoke_with_backoff(self, prompt):
        for attempt in range(6):
            try:
                return await self.model.ainvoke(prompt)
            except Exception as exc:
                if attempt == 5 or not self._retryable(exc):
                    raise
                await asyncio.sleep(min(60, 2 ** attempt + random.random()))

    def generate(self, prompt: str, schema: BaseModel | None = None):
        if schema is None:
            return _text(self._invoke_with_backoff(prompt))
        request = prompt + self._json_hint(schema)
        for attempt in range(2):
            try:
                return self._parse(_text(self._invoke_with_backoff(request)), schema)
            except Exception:
                if attempt:
                    raise
                print("Judge returned invalid JSON; retrying once with a repair instruction.")
                request += "\\nYour previous reply was not valid JSON for the schema. Reply with ONLY the JSON object."

    async def a_generate(self, prompt: str, schema: BaseModel | None = None):
        if schema is None:
            return _text(await self._ainvoke_with_backoff(prompt))
        request = prompt + self._json_hint(schema)
        for attempt in range(2):
            try:
                return self._parse(_text(await self._ainvoke_with_backoff(request)), schema)
            except Exception:
                if attempt:
                    raise
                print("Judge returned invalid JSON; retrying once with a repair instruction.")
                request += "\\nYour previous reply was not valid JSON for the schema. Reply with ONLY the JSON object."

    def get_model_name(self):
        return self.model_name




@lru_cache(maxsize=1)
def get_judge():
    from evals.config import JUDGE_MODEL
    return OpenRouterJudge(JUDGE_MODEL)
