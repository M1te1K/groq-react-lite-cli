import json
import logging
import os
import re
from collections.abc import Callable
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from react_lite.console import describe_action, describe_tool_error


class Events:
    def __init__(
        self,
        logger: logging.Logger,
        secret: str = "",
        trace: Callable[[str], None] | None = None,
    ):
        self.logger = logger
        self.secret = secret
        self.trace = trace
        pattern = r"gsk_[A-Za-z0-9_-]+"
        if secret:
            pattern += "|" + re.escape(secret)
        self._secret_pattern = re.compile(pattern)

    def _redact_values(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._redact_values(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._redact_values(item) for item in value]
        if value is None or isinstance(value, (bool, int, float)):
            return value
        # Маскируем также значения, которые раньше сериализовались через default=str.
        return self._secret_pattern.sub("[REDACTED]", str(value))

    def emit(self, event: str, request_id: str, **fields: Any) -> None:
        record = {
            "time": datetime.now(UTC).isoformat(),
            "user_id": "local",
            "request_id": request_id,
            "event": event,
            **fields,
        }
        safe = self._redact_values(record)
        line = json.dumps(safe, ensure_ascii=False)
        self.logger.info(line)
        message = None
        if self.trace:
            if event == "tool_call" and fields.get("will_execute", True):
                message = describe_action(fields["tool"], fields["arguments"])
            elif event == "step" and isinstance(fields.get("Action"), dict):
                if "tool" in fields["Action"]:
                    message = describe_tool_error(fields.get("Observation"))
            elif event == "retry":
                message = (
                    "ЦБ не ответил, пробую ещё раз…"
                    if fields["service"] == "cbr"
                    else "Повторяю обращение к сервису…"
                )
            if message:
                self.trace(self._redact_values(message))


def file_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    os.close(descriptor)
    logger = logging.getLogger("react_lite")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    # Логируем категории ошибок сами, без HTTP-заголовков и тел SDK-исключений.
    for name in ("groq", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.CRITICAL + 1)
    return logger
