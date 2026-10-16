"""Оформление однозначных валютных результатов без повторов текста модели."""

import json
import re
from datetime import date
from decimal import Decimal, localcontext
from typing import Any


def _results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [json.loads(message["content"]) for message in messages if message["role"] == "tool"]


def _number(value: str, money: bool = False) -> str:
    with localcontext() as context:
        context.prec = 110
        text = format(Decimal(value), ",.2f" if money else "f")
    if not money and "." in text:
        text = text.rstrip("0").rstrip(".")
    return text.replace(",", " ").replace(".", ",")


def format_final(
    answer: str,
    current: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> str:
    results = _results(current)
    if not results or any(result.get("ok") is not True for result in results):
        return answer
    rates = {
        result["currency"]: result
        for result in _results(history)
        if result.get("ok") and "rate_rub_per_unit" in result
    }
    mentioned = [
        rate
        for currency, rate in rates.items()
        if re.search(r"\b" + re.escape(currency) + r"\b", answer, re.IGNORECASE)
    ]
    if len(mentioned) != 1:
        return answer
    rate = mentioned[0]
    current_rates = [result for result in results if "rate_rub_per_unit" in result]
    calculations = [result for result in results if "operation" in result]
    prefix = ""
    if len(calculations) == 1 and len(current_rates) <= 1 and results[-1] == calculations[0]:
        calculation = calculations[0]
        if calculation["operation"] != "multiply":
            return answer
        rate_value = Decimal(rate["rate_rub_per_unit"])
        if Decimal(calculation["b"]) == rate_value:
            amount = calculation["a"]
        elif Decimal(calculation["a"]) == rate_value:
            amount = calculation["b"]
        else:
            return answer
        prefix = (
            f"{_number(amount)} {rate['currency']} = "
            f"{_number(calculation['result'], money=True)} ₽. "
        )
    elif calculations or current_rates != [rate]:
        # Обычные вопросы, ошибки и составные задачи сохраняют ответ модели.
        return answer
    effective_date = date.fromisoformat(rate["effective_date"]).strftime("%d.%m.%Y")
    return (
        f"{prefix}Курс ЦБ: {_number(rate['rate_rub_per_unit'])} ₽ за 1 {rate['currency']}, "
        f"дата курса: {effective_date}; источник: Банк России ({rate['source_url']})."
    )
