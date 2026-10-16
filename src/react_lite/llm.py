from dataclasses import dataclass, field
from typing import Any, Protocol

import groq

from react_lite.budget import Budget


class LLMUnavailable(Exception):
    def __init__(self, category: str, status: int | None = None):
        self.category, self.status = category, status
        super().__init__(category)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str

    def as_message(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


@dataclass(frozen=True)
class Reply:
    content: str | None = None
    calls: list[ToolCall] = field(default_factory=list)

    def as_message(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.calls:
            result["tool_calls"] = [call.as_message() for call in self.calls]
        return result


class LanguageModel(Protocol):
    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], budget: Budget
    ) -> Reply: ...


class GroqModel:
    def __init__(
        self, client: groq.Groq | None, model: str, timeout: float = 10, attempts: int = 2
    ):
        self.client, self.model = client, model
        self.timeout, self.attempts = timeout, attempts

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], budget: Budget
    ) -> Reply:
        if self.client is None:
            raise LLMUnavailable("missing_api_key")
        for attempt in range(self.attempts):
            try:
                response = self.client.with_options(
                    max_retries=0, timeout=budget.timeout(self.timeout)
                ).chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    tools=tools,  # type: ignore[arg-type]
                    tool_choice="auto",
                    parallel_tool_calls=False,
                    temperature=0,
                    max_completion_tokens=4096,
                )
                budget.remaining()
                if not response.choices or response.choices[0].finish_reason not in {
                    "stop",
                    "tool_calls",
                }:
                    raise LLMUnavailable("incomplete_response")
                message = response.choices[0].message
                calls = message.tool_calls or []
                if message.content is not None and not isinstance(message.content, str):
                    raise LLMUnavailable("invalid_response")
                for call in calls:
                    if call.type != "function" or not all(
                        isinstance(value, str)
                        for value in (call.id, call.function.name, call.function.arguments)
                    ):
                        raise LLMUnavailable("invalid_response")
                if response.choices[0].finish_reason == "tool_calls" and not calls:
                    raise LLMUnavailable("invalid_response")
                # reasoning и другие служебные поля никогда не выводим и не логируем.
                return Reply(
                    message.content,
                    [
                        ToolCall(call.id, call.function.name, call.function.arguments)
                        for call in calls
                    ],
                )
            except groq.APIError as exc:
                budget.remaining()
                status = getattr(exc, "status_code", None)
                retryable = (
                    isinstance(exc, groq.APIConnectionError)
                    or status == 429
                    or (status is not None and status >= 500)
                )
                if not retryable or attempt + 1 == self.attempts:
                    # Не передаём текст API-ошибки: он может содержать секреты/запрос.
                    raise LLMUnavailable(type(exc).__name__, status) from None
                budget.retry("groq", attempt + 2)
                budget.backoff(attempt)
            except (ValueError, TypeError, AttributeError) as exc:
                raise LLMUnavailable("invalid_response") from exc
        raise AssertionError("unreachable")
