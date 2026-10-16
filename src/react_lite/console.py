"""Короткие сообщения для диалога; технические данные остаются в журнале."""

import os
import sys
from typing import Any, TextIO

from pydantic import ValidationError

from react_lite.tools import CalculateArgs, RateArgs


def describe_action(tool: str, arguments: str) -> str | None:
    if tool not in {"get_exchange_rate", "calculate"}:
        return None
    fallback = (
        "Проверяю данные для получения курса…"
        if tool == "get_exchange_rate"
        else "Проверяю данные для расчёта…"
    )
    if len(arguments) > 4096:
        return fallback
    try:
        if tool == "get_exchange_rate":
            args = RateArgs.model_validate_json(arguments)
            return f"Получаю курс {args.currency} в ЦБ…"
        args = CalculateArgs.model_validate_json(arguments)
        symbol = {"add": "+", "subtract": "−", "multiply": "×", "divide": "÷"}[args.operation]
        a, b = args.a.replace(".", ","), args.b.replace(".", ",")
        if len(a) + len(b) > 48:
            return "Выполняю расчёт…"
        return f"Рассчитываю {a} {symbol} {b}…"
    except (ValidationError, ValueError, RecursionError):
        return fallback


def describe_tool_error(observation: Any) -> str | None:
    if not isinstance(observation, dict) or observation.get("ok") is not False:
        return None
    return {
        "cbr_unavailable": "Не удалось получить курс ЦБ.",
        "invalid_cbr_data": "ЦБ вернул некорректные данные курса.",
        "unknown_currency": "В ЦБ нет курса этой валюты.",
        "invalid_arguments": "Не удалось выполнить действие: проверьте данные.",
        "unknown_tool": "Выбранное действие недоступно.",
        "division_by_zero": "Не удалось посчитать: деление на ноль.",
    }.get(observation.get("error", {}).get("code"))


class ActionPrinter:
    def __init__(self, stream: TextIO | None = None):
        self.stream = stream

    def __call__(self, message: str) -> None:
        stream = self.stream if self.stream is not None else sys.stdout
        line = f"→ {message}"
        if stream.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb":
            line = f"\033[36m{line}\033[0m"
        print(line, file=stream, flush=True)
