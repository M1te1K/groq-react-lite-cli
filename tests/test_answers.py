import json

from react_lite.answers import format_final


def tool(result):
    return {"role": "tool", "content": json.dumps(result)}


RATE = {
    "ok": True,
    "currency": "EUR",
    "rate_rub_per_unit": "92.1234",
    "effective_date": "2026-10-15",
    "source_url": "https://www.cbr.ru/test",
}
CALC = {"ok": True, "operation": "multiply", "a": "500", "b": "92.1234", "result": "46061.7000"}


def test_conversion_uses_observations_once_even_if_model_repeats():
    current = [tool(RATE), tool(CALC)]
    answer = format_final(
        "500 EUR = ... effective_date... 500 EUR = ... source_url...", current, current
    )
    assert answer == (
        "500 EUR = 46 061,70 ₽. Курс ЦБ: 92,1234 ₽ за 1 EUR, дата курса: "
        "15.10.2026; источник: Банк России (https://www.cbr.ru/test)."
    )


def test_follow_up_uses_previous_rate_and_new_calculation():
    current = [tool({**CALC, "a": "1000", "result": "92123.4000"})]
    answer = format_final("EUR ... EUR ...", current, [tool(RATE), *current])
    assert answer.startswith("1000 EUR = 92 123,40 ₽.")
    assert answer.count("источник:") == 1


def test_rate_only_has_friendly_labels_and_actual_date():
    current = [tool(RATE)]
    answer = format_final("EUR effective_date source_url", current, current)
    assert answer.startswith("Курс ЦБ: 92,1234 ₽ за 1 EUR, дата курса: 15.10.2026;")
    assert "effective_date" not in answer and "source_url" not in answer


def test_does_not_rewrite_ordinary_errors_unrelated_or_multiple_calculations():
    for current in (
        [],
        [tool({"ok": False})],
        [tool({**CALC, "b": "2"})],
        [tool(CALC), tool(CALC)],
    ):
        assert format_final("Ответ EUR", current, [tool(RATE), *current]) == "Ответ EUR"
    assert format_final("Просто арифметика", [tool(CALC)], [tool(RATE), tool(CALC)]) == (
        "Просто арифметика"
    )


def test_reverse_operands_and_rounding_use_decimal():
    calculation = {**CALC, "a": "92.1234", "b": "0.5", "result": "46.0617"}
    answer = format_final("0.5 EUR", [tool(calculation)], [tool(RATE), tool(calculation)])
    assert answer.startswith("0,5 EUR = 46,06 ₽.")
