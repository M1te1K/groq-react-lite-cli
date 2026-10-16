import json
from decimal import Decimal

import httpx
import pytest
from pydantic import ValidationError

from react_lite.budget import Budget
from react_lite.tools import CalculateArgs, ExchangeRates, RateArgs, calculate, parse_rate
from tests.conftest import XML


@pytest.mark.parametrize(
    ("operation", "a", "b", "expected"),
    [
        ("add", "0.1", "0.2", "0.3"),
        ("subtract", "0.3", "0.1", "0.2"),
        ("multiply", "500", "90.1234", "45061.7000"),
        ("divide", "10", "4", "2.5"),
        (
            "multiply",
            "999999999999999999999999999999.12345678901234567890",
            "999999999999999999999999999999.12345678901234567890",
            None,
        ),
    ],
)
def test_decimal_arithmetic(operation, a, b, expected):
    result = calculate(CalculateArgs(operation=operation, a=a, b=b))
    assert result["ok"]
    if expected:
        assert result["result"] == expected
    else:
        from decimal import localcontext

        with localcontext() as ctx:
            ctx.prec = 120
            assert Decimal(result["result"]) == Decimal(a) * Decimal(b)


def test_division_by_zero():
    result = calculate(CalculateArgs(operation="divide", a="1", b="0.00"))
    assert result["error"]["code"] == "division_by_zero"


@pytest.mark.parametrize(
    "bad",
    [
        0.1,
        500,
        True,
        None,
        "NaN",
        "Infinity",
        "1e3",
        "0,1",
        "__import__('os').system('whoami')",
        "1" * 31,
    ],
)
def test_reject_inexact_or_invalid_numbers(bad):
    with pytest.raises(ValidationError):
        CalculateArgs(operation="add", a=bad, b="1")


def test_rate_nominal_and_actual_date():
    result = parse_rate(XML, RateArgs(currency="JPY", date="2026-10-16"))
    assert result["ok"]
    assert result["rate_rub_per_unit"] == "0.6250"
    assert result["nominal"] == "100"
    assert result["requested_date"] == "2026-10-16"
    assert result["effective_date"] == "2026-10-15"
    assert "date_req=16%2F10%2F2026" in result["source_url"]
    assert result["source"].startswith("Банк России")


def test_unknown_currency():
    assert parse_rate(XML, RateArgs(currency="ZZZ", date="2026-10-16"))["error"]["code"] == (
        "unknown_currency"
    )


def test_bad_encoding_and_deep_arguments_do_not_crash(make_agent):
    body = b'<?xml version="1.0" encoding="made-up"?><ValCurs Date="15.10.2026"/>'
    assert parse_rate(body, RateArgs(currency="USD", date="2026-10-16"))["ok"] is False
    agent, _ = make_agent([])
    result = agent.tools.execute("calculate", "[" * 1500 + "]" * 1500, Budget(2))
    assert result["error"]["code"] == "invalid_arguments"


def test_cbr_404_is_not_retried():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = ExchangeRates(client, attempts=3).get(
            RateArgs(currency="USD", date="2026-10-16"),
            Budget(2),
        )
    assert len(seen) == 1
    assert result["error"]["code"] == "cbr_unavailable"


@pytest.mark.parametrize(
    "body",
    [
        b"<html>unavailable</html>",
        b"<broken",
        b'<ValCurs Date="not a date"/>',
        XML.replace(b"100</Nominal>", b"0</Nominal>"),
        XML.replace(b"62,5000", b"NaN"),
        XML.replace(b"62,5000", b"-10"),
        XML.replace(b"15.10.2026", b"17.10.2026"),
        b'<!DOCTYPE x [<!ENTITY x "boom">]><ValCurs Date="15.10.2026"/>',
    ],
)
def test_bad_cbr_xml(body):
    assert parse_rate(body, RateArgs(currency="JPY", date="2026-10-16"))["error"]["code"] == (
        "invalid_cbr_data"
    )


def test_cbr_request_passes_date():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, content=XML)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = ExchangeRates(client).get(RateArgs(currency="USD", date="2026-10-16"), Budget(2))
    assert result["ok"]
    assert seen[0].url.params["date_req"] == "16/10/2026"
    assert seen[0].url.host == "www.cbr.ru"


@pytest.mark.parametrize("failure", [503, 429, "timeout"])
def test_cbr_retries_bounded(failure, monkeypatch):
    seen = []
    monkeypatch.setattr(Budget, "backoff", lambda *args: None)

    def handler(request):
        seen.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout")
        return httpx.Response(failure)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = ExchangeRates(client, attempts=2).get(
            RateArgs(currency="USD", date="2026-10-16"),
            Budget(2),
        )
    assert len(seen) == 2
    assert result["error"]["code"] == "cbr_unavailable"


@pytest.mark.parametrize(
    ("name", "payload", "code"),
    [
        ("execute_code", "{}", "unknown_tool"),
        ("calculate", '{"operation":"add","a":0.1,"b":"1"}', "invalid_arguments"),
        ("calculate", '{"operation":"pow","a":"1","b":"2"}', "invalid_arguments"),
        ("calculate", '{"operation":"add","a":"1","b":"2","extra":true}', "invalid_arguments"),
        ("calculate", "[1,2]", "invalid_arguments"),
        ("calculate", "not json", "invalid_arguments"),
        ("get_exchange_rate", '{"currency":"USD","date":"2026-02-30"}', "invalid_arguments"),
        ("get_exchange_rate", '{"currency":"usd","date":"today"}', "invalid_arguments"),
    ],
)
def test_tool_validation(make_agent, name, payload, code):
    agent, _ = make_agent([])
    result = agent.tools.execute(name, payload, Budget(2))
    assert result["error"]["code"] == code
    assert json.loads(json.dumps(result)) == result
