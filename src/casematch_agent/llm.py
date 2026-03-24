from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol
from urllib import error, request


class JsonLLMClient(Protocol):
    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> dict[str, Any]:
        ...


class LLMClientError(RuntimeError):
    pass


@dataclass
class OpenAICompatibleConfig:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: int = 30
    use_json_mode: bool = True

    @classmethod
    def from_env(cls) -> "OpenAICompatibleConfig":
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini",
            base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").strip() or "https://api.openai.com/v1",
            timeout_seconds=int(os.getenv("OPENAI_TIMEOUT_SECONDS", "30")),
            use_json_mode=os.getenv("OPENAI_USE_JSON_MODE", "1").strip() != "0",
        )

    def is_enabled(self) -> bool:
        return bool(self.api_key)


@dataclass
class OpenAICompatibleClient:
    config: OpenAICompatibleConfig

    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }

        if self.config.use_json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response_json = self._post(payload)
        except LLMClientError as exc:
            if not self.config.use_json_mode:
                raise
            message = str(exc).lower()
            if "response_format" not in message and "json_object" not in message:
                raise
            payload.pop("response_format", None)
            response_json = self._post(payload)

        content = self._extract_content(response_json)
        return self._extract_json(content)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMClientError(detail or str(exc)) from exc
        except error.URLError as exc:
            raise LLMClientError(str(exc)) from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMClientError(f"Invalid JSON response: {body}") from exc

    def _extract_content(self, response_json: dict[str, Any]) -> str:
        choices = response_json.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMClientError(f"Unexpected response shape: {response_json}")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            return "\n".join(parts).strip()
        if not isinstance(content, str):
            raise LLMClientError(f"Unexpected content type: {content}")
        return content.strip()

    def _extract_json(self, content: str) -> dict[str, Any]:
        trimmed = content.strip()
        if trimmed.startswith("```"):
            trimmed = trimmed.strip("`")
            if trimmed.startswith("json"):
                trimmed = trimmed[4:].strip()
        try:
            parsed = json.loads(trimmed)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        start = trimmed.find("{")
        end = trimmed.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise LLMClientError(f"Model did not return a JSON object: {content}")
        try:
            parsed = json.loads(trimmed[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMClientError(f"Failed to parse JSON content: {content}") from exc
        if not isinstance(parsed, dict):
            raise LLMClientError(f"JSON content is not an object: {content}")
        return parsed
