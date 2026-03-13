from __future__ import annotations

import json
from typing import Any, Optional

import httpx

from ..config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_REASONING_EFFORT, OPENAI_TIMEOUT_SECONDS


def _coerce_text(value: Any) -> str:
    return str(value or "").strip()


class OpenAIEditorialClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.api_key = _coerce_text(api_key or OPENAI_API_KEY)
        self.model = _coerce_text(model or OPENAI_MODEL) or "gpt-5.4"
        self.base_url = _coerce_text(base_url or OPENAI_BASE_URL) or "https://api.openai.com/v1"
        self.timeout_seconds = timeout_seconds or OPENAI_TIMEOUT_SECONDS

    def available(self) -> bool:
        return bool(self.api_key)

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.available(),
            "model": self.model,
            "base_url": self.base_url,
        }

    def create_text(
        self,
        *,
        instructions: str,
        input_text: str,
        max_output_tokens: int,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        payload = {
            "model": self.model,
            "instructions": instructions,
            "input": input_text,
            "reasoning": {"effort": reasoning_effort or OPENAI_REASONING_EFFORT},
            "max_output_tokens": max_output_tokens,
            "store": False,
        }
        response = self._request(payload)
        text = self._extract_output_text(response)
        if not text:
            raise RuntimeError("OpenAI response did not include output_text")
        return text

    def create_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict[str, Any],
        max_output_tokens: int,
        reasoning_effort: Optional[str] = None,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "instructions": instructions,
            "input": input_text,
            "reasoning": {"effort": reasoning_effort or OPENAI_REASONING_EFFORT},
            "max_output_tokens": max_output_tokens,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        response = self._request(payload)
        text = self._extract_output_text(response)
        if not text:
            raise RuntimeError("OpenAI structured response was empty")
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise RuntimeError("OpenAI structured response was not an object")
        return parsed

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.available():
            raise RuntimeError("OPENAI_API_KEY is not configured")
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/responses",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    def _extract_output_text(self, payload: dict[str, Any]) -> str:
        direct = _coerce_text(payload.get("output_text"))
        if direct:
            return direct
        output = payload.get("output") or []
        parts: list[str] = []
        for item in output:
            if item.get("type") != "message":
                continue
            for content_item in item.get("content") or []:
                content_type = content_item.get("type")
                if content_type in {"output_text", "text"}:
                    text = _coerce_text(content_item.get("text"))
                    if text:
                        parts.append(text)
        return "\n".join(part for part in parts if part).strip()
