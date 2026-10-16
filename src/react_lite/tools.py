import json
import re
import xml.etree.ElementTree as ET
from datetime import date as Date
from datetime import datetime
from decimal import Decimal, DecimalException, localcontext
from typing import Annotated, Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from react_lite.budget import Budget

CBR_URL = "https://www.cbr.ru/scripts/XML_daily.asp"
DECIMAL_PATTERN = r"^-?(?:0|[1-9][0-9]{0,29})(?:\.[0-9]{1,20})?$"
DecimalString = Annotated[str, Field(strict=True, pattern=DECIMAL_PATTERN, max_length=52)]


class CalculateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["add", "subtract", "multiply", "divide"]
    a: DecimalString
    b: DecimalString


class RateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    currency: str = Field(strict=True, pattern=r"^[A-Z]{3}$")
    date: str = Field(strict=True, pattern=r"^\d{4}-\d{2}-\d{2}$")

    @field_validator("date")
    @classmethod
    def valid_date(cls, value: str) -> str:
        Date.fromisoformat(value)
        return value


def error(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}}


def decimal_text(value: Decimal) -> str:
    return format(value, "f")


def calculate(args: CalculateArgs) -> dict[str, Any]:
    a, b = Decimal(args.a), Decimal(args.b)
    if args.operation == "divide" and b == 0:
        return error("division_by_zero", "Деление на ноль невозможно")
    with localcontext() as context:
        # До 30 целых и 20 дробных цифр на операнд: произведение помещается точно.
        context.prec = 110
        if args.operation == "add":
            result = a + b
        elif args.operation == "subtract":
            result = a - b
        elif args.operation == "multiply":
            result = a * b
        else:
            result = a / b
    return {
        "ok": True,
        "operation": args.operation,
        "a": args.a,
        "b": args.b,
        "result": decimal_text(result),
    }


def parse_rate(xml: bytes, args: RateArgs) -> dict[str, Any]:
    try:
        if len(xml) > 2_000_000 or b"<!DOCTYPE" in xml.upper() or b"<!ENTITY" in xml.upper():
            raise ValueError("Unsupported XML")
        root = ET.fromstring(xml)
        if root.tag != "ValCurs":
            raise ValueError("Unexpected root")
        effective = datetime.strptime(root.attrib["Date"], "%d.%m.%Y").date()
        if effective > Date.fromisoformat(args.date):
            raise ValueError("Future effective date")
        for item in root.findall("Valute"):
            if item.findtext("CharCode") != args.currency:
                continue
            nominal_text = item.findtext("Nominal", "")
            value_text = item.findtext("Value", "").replace(",", ".")
            if not re.fullmatch(r"[1-9][0-9]{0,8}", nominal_text):
                raise ValueError("Invalid nominal")
            if not re.fullmatch(r"[0-9]{1,15}(?:\.[0-9]{1,15})?", value_text):
                raise ValueError("Invalid value")
            nominal, value = Decimal(nominal_text), Decimal(value_text)
            if value <= 0:
                raise ValueError("Invalid value")
            with localcontext() as context:
                context.prec = 60
                rate = value / nominal
            return {
                "ok": True,
                "currency": args.currency,
                "base_currency": "RUB",
                "requested_date": args.date,
                "effective_date": effective.isoformat(),
                "nominal": nominal_text,
                "value_rub": decimal_text(value),
                "rate_rub_per_unit": decimal_text(rate),
                "source": "Банк России (официальный курс ЦБ)",
                "source_url": str(
                    httpx.URL(
                        CBR_URL,
                        params={
                            "date_req": Date.fromisoformat(args.date).strftime("%d/%m/%Y"),
                        },
                    )
                ),
            }
        return error("unknown_currency", f"Код {args.currency} отсутствует в данных ЦБ")
    except (ET.ParseError, KeyError, LookupError, ValueError, DecimalException):
        return error("invalid_cbr_data", "ЦБ вернул некорректные данные курса")


class ExchangeRates:
    def __init__(self, client: httpx.Client, timeout: float = 10, attempts: int = 2):
        self.client, self.timeout, self.attempts = client, timeout, attempts

    def get(self, args: RateArgs, budget: Budget) -> dict[str, Any]:
        params = {"date_req": Date.fromisoformat(args.date).strftime("%d/%m/%Y")}
        for attempt in range(self.attempts):
            try:
                # Ограничиваем размер до загрузки всего тела; дедлайн прерывает и slow read.
                with self.client.stream(
                    "GET", CBR_URL, params=params, timeout=budget.timeout(self.timeout)
                ) as response:
                    response.raise_for_status()
                    chunks, size = [], 0
                    for chunk in response.iter_bytes():
                        budget.remaining()
                        size += len(chunk)
                        if size > 2_000_000:
                            return error("invalid_cbr_data", "Ответ ЦБ превышает допустимый размер")
                        chunks.append(chunk)
                return parse_rate(b"".join(chunks), args)
            except httpx.HTTPError as exc:
                budget.remaining()
                retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code == 429 or exc.response.status_code >= 500
                )
                if not retryable or attempt + 1 == self.attempts:
                    return error("cbr_unavailable", "Не удалось получить официальный курс ЦБ")
                budget.retry("cbr", attempt + 2)
                budget.backoff(attempt)
        raise AssertionError("unreachable")


class Tools:
    def __init__(self, rates: ExchangeRates):
        self.rates = rates

    @staticmethod
    def schemas() -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_exchange_rate",
                    "description": (
                        "Получить официальный курс валюты к RUB из ЦБ на конкретную дату. "
                        "Используй rate_rub_per_unit для расчёта, effective_date для даты курса."
                    ),
                    "parameters": RateArgs.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": (
                        "Выполнить арифметику через Decimal. a и b — строки с десятичной точкой, "
                        "до 30 целых и 20 дробных цифр, без экспоненты. "
                        "Для конвертации передай сумму и уже полученный rate_rub_per_unit."
                    ),
                    "parameters": CalculateArgs.model_json_schema(),
                },
            },
        ]

    def execute(self, name: str, arguments: str, budget: Budget) -> dict[str, Any]:
        if name not in {"get_exchange_rate", "calculate"}:
            return error("unknown_tool", "Такого инструмента нет")
        if len(arguments) > 4096:
            return error("invalid_arguments", "Аргументы превышают допустимый размер")
        try:
            # JSON-числа запрещены схемой: деньги всегда приходят строками, без float.
            payload = json.loads(arguments)
            if name == "calculate":
                return calculate(CalculateArgs.model_validate(payload))
            args = RateArgs.model_validate(payload)
            return self.rates.get(args, budget)
        except (ValidationError, ValueError, DecimalException, RecursionError):
            return error(
                "invalid_arguments",
                "Аргументы не соответствуют схеме инструмента. "
                "Проверь имя операции, строковые числа, код валюты и дату YYYY-MM-DD.",
            )
