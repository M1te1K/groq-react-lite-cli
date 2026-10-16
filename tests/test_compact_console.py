import io
import json
import logging
from uuid import uuid4

import groq
import httpx
import pytest

from react_lite.agent import TOOL_LIMIT
from react_lite.budget import Budget
from react_lite.cli import interact, main
from react_lite.console import ActionPrinter
from react_lite.events import Events
from react_lite.llm import GroqModel, Reply, ToolCall
from tests.test_agent import call
from tests.test_llm_cli_events import completion

TECHNICAL = (
    "Thought",
    "Action:",
    "Observation",
    "chat.completions",
    "tool_call_id",
    "rate_rub_per_unit",
    "effective_date",
    "source_url",
    '"operation"',
    "{",
    "}",
)


def recording_events(console, secret=""):
    log = io.StringIO()
    logger = logging.getLogger(f"compact.{uuid4().hex}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(logging.StreamHandler(log))
    return Events(logger, secret, ActionPrinter(console)), log


def test_steps_precede_tools_and_only_final_is_printed_once(make_agent, monkeypatch):
    final = (
        "500 USD = 45 061,70 ₽. Курс ЦБ: 90,1234 ₽ за 1 USD, дата курса: 15.10.2026; "
        "источник: Банк России (https://www.cbr.ru/scripts/XML_daily.asp?date_req=16%2F10%2F2026)."
    )
    agent, _ = make_agent(
        [
            call("get_exchange_rate", {"currency": "USD", "date": "2026-10-16"}, "rate"),
            call("calculate", {"operation": "multiply", "a": "500", "b": "90.1234"}, "calc"),
            Reply(final),
            Reply("Привет!"),
        ]
    )
    console = io.StringIO()
    agent.events, log = recording_events(console)
    execute = agent.tools.execute
    seen = []

    def checked_execute(name, arguments, budget):
        seen.append(name)
        lines = console.getvalue().splitlines()
        expected = (
            "→ Получаю курс USD в ЦБ…"
            if name == "get_exchange_rate"
            else "→ Рассчитываю 500 × 90,1234…"
        )
        assert lines[-1] == expected  # Уже выведено, инструмент ещё не выполнялся.
        assert console.getvalue().count("→") == len(seen)
        assert final not in console.getvalue()
        return execute(name, arguments, budget)

    monkeypatch.setattr(agent.tools, "execute", checked_execute)
    inputs = iter(["Посчитай 500 USD по курсу ЦБ", "Привет", "/exit"])
    interact(agent, lambda p: next(inputs), lambda s: print(s, file=console))
    shown = console.getvalue()
    assert seen == ["get_exchange_rate", "calculate"]
    assert shown.count("→") == 2  # Приветствие без фиктивных действий.
    assert shown.count(final) == 1
    assert shown.count("Бот: Привет!") == 1
    assert all(marker not in shown for marker in TECHNICAL)
    assert "\033[" not in shown
    rows = [json.loads(line) for line in log.getvalue().splitlines()]
    assert all({"time", "user_id", "request_id"} <= row.keys() for row in rows)
    steps = [row for row in rows if row["event"] == "step"]
    assert all({"Thought", "Action", "Observation"} <= row.keys() for row in steps)
    assert [row["tool"] for row in rows if row["event"] == "tool_call"] == seen
    assert any(
        row.get("Observation", {}).get("result") == "45061.7000"
        for row in steps
        if isinstance(row["Observation"], dict)
    )
    assert any(
        row["Action"].get("tool_call_id") == "calc" and '"500"' in row["Action"]["arguments"]
        for row in steps
        if isinstance(row["Action"], dict) and "tool" in row["Action"]
    )


@pytest.mark.parametrize("argv", [[], ["--trace"]])
def test_full_main_uses_compact_steps_in_both_modes(argv, tmp_path, monkeypatch, capsys):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        if len(seen) == 1:
            assert "служебные имена никогда не выводи" in seen[0]["messages"][0]["content"]
            response = completion(
                None,
                [
                    {
                        "id": "internal_calc",
                        "type": "function",
                        "function": {
                            "name": "calculate",
                            "arguments": '{"operation":"add","a":"2","b":"3"}',
                        },
                    }
                ],
                "tool_calls",
            )
        else:
            response = completion("2 + 3 = 5")
        return httpx.Response(200, json=response)

    real_groq = groq.Groq
    monkeypatch.setattr(
        "react_lite.cli.groq.Groq",
        lambda **kwargs: real_groq(
            **kwargs,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "test-compact")
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "agent.jsonl"))
    monkeypatch.setattr("sys.stdin", io.StringIO("Сложи 2 и 3\n/exit\n"))
    logger = logging.getLogger("react_lite")
    try:
        assert main(argv) == 0
    finally:
        for log_handler in logger.handlers[:]:
            log_handler.close()
            logger.removeHandler(log_handler)
    shown = capsys.readouterr()
    assert not shown.err
    assert shown.out.count("→ Рассчитываю 2 + 3…") == 1
    assert shown.out.count("2 + 3 = 5") == 1
    assert "internal_calc" not in shown.out
    assert all(marker not in shown.out for marker in TECHNICAL)
    rows = [json.loads(line) for line in (tmp_path / "agent.jsonl").read_text().splitlines()]
    assert any(row.get("Action") == "final" for row in rows)
    assert any(row.get("tool_call_id") == "internal_calc" for row in rows)


class TerminalBuffer(io.StringIO):
    def __init__(self, tty):
        super().__init__()
        self.tty = tty
        self.flushed = False

    def isatty(self):
        return self.tty

    def flush(self):
        self.flushed = True


@pytest.mark.parametrize(
    ("tty", "no_color", "term", "colored"),
    [
        (True, None, "xterm", True),
        (False, None, "xterm", False),
        (True, "", "xterm", False),
        (True, "1", "xterm", False),
        (False, "1", "xterm", False),
        (True, None, "dumb", False),
    ],
)
def test_color_only_for_tty_and_without_no_color(tty, no_color, term, colored, monkeypatch):
    if no_color is None:
        monkeypatch.delenv("NO_COLOR", raising=False)
    else:
        monkeypatch.setenv("NO_COLOR", no_color)
    monkeypatch.setenv("TERM", term)
    console = TerminalBuffer(tty)
    ActionPrinter(console)("Выполняю расчёт…")
    line = console.getvalue()
    assert console.flushed  # Сообщение доступно до завершения инструмента.
    assert ("\033[36m" in line) is colored
    assert line == ("\033[36m→ Выполняю расчёт…\033[0m\n" if colored else "→ Выполняю расчёт…\n")


def test_no_progress_for_rejected_batch(make_agent):
    batch = Reply(calls=[ToolCall(f"c{i}", "calculate", "{}") for i in range(2)])
    agent, _ = make_agent([batch], max_tool_calls=1)
    console = io.StringIO()
    agent.events, _ = recording_events(console)
    assert agent.run("Считай") == TOOL_LIMIT
    assert console.getvalue() == ""  # Ни один вызов не выполнялся.


@pytest.mark.parametrize("secret", ["USD", "Рассчитываю", "step", "Action", "gsk_FakeKey"])
def test_compact_steps_mask_keys_and_keep_jsonl_fields(secret):
    console = io.StringIO()
    events, log = recording_events(console, secret)
    events.emit(
        "tool_call",
        "r1",
        step=1,
        tool="get_exchange_rate",
        arguments='{"currency":"USD","date":"2026-10-16"}',
        tool_call_id="id",
    )
    events.emit(
        "tool_call",
        "r1",
        step=2,
        tool="calculate",
        arguments='{"operation":"add","a":"2","b":"3"}',
        tool_call_id="id2",
    )
    assert secret not in console.getvalue()
    assert "gsk_" not in console.getvalue()
    assert all("step" in json.loads(line) for line in log.getvalue().splitlines())


def test_tool_error_and_cbr_retry_are_short_and_logged(make_agent, monkeypatch):
    count = 0

    def unavailable(request):
        nonlocal count
        count += 1
        return httpx.Response(503)

    agent, _ = make_agent(
        [
            call("get_exchange_rate", {"currency": "EUR", "date": "2026-10-16"}),
            Reply("Попробуйте запросить курс позднее."),
        ],
        handler=unavailable,
    )
    agent.tools.rates.attempts = 2
    console = io.StringIO()
    agent.events, log = recording_events(console)
    monkeypatch.setattr(Budget, "backoff", lambda *a: None)
    assert agent.run("Курс EUR?") == "Попробуйте запросить курс позднее."
    assert count == 2
    assert console.getvalue().splitlines() == [
        "→ Получаю курс EUR в ЦБ…",
        "→ ЦБ не ответил, пробую ещё раз…",
        "→ Не удалось получить курс ЦБ.",
    ]
    assert all(marker not in console.getvalue() for marker in TECHNICAL)
    rows = [json.loads(line) for line in log.getvalue().splitlines()]
    assert any(row["event"] == "retry" and row["attempt"] == 2 for row in rows)
    assert any(
        row.get("Observation", {}).get("error", {}).get("code") == "cbr_unavailable"
        for row in rows
        if isinstance(row.get("Observation"), dict)
    )


def test_groq_retry_does_not_invent_tool_step(make_agent, monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        return (
            httpx.Response(503)
            if len(seen) == 1
            else httpx.Response(200, json=completion("Привет!"))
        )

    monkeypatch.setattr(Budget, "backoff", lambda *a: None)
    with groq.Groq(
        api_key="test",
        http_client=httpx.Client(
            transport=httpx.MockTransport(handler),
        ),
    ) as client:
        agent, _ = make_agent([])
        agent.model = GroqModel(client, "test")
        console = io.StringIO()
        agent.events, _ = recording_events(console)
        assert agent.run("Привет") == "Привет!"
    assert console.getvalue() == "→ Повторяю обращение к сервису…\n"


def test_invalid_arguments_and_division_error_stay_readable(make_agent):
    agent, _ = make_agent(
        [
            call("calculate", {"operation": "add", "a": "gsk_DoNotShow", "b": "2"}, "bad"),
            call("calculate", {"operation": "divide", "a": "1", "b": "0"}, "zero"),
            Reply("Уточните числа для расчёта."),
        ]
    )
    console = io.StringIO()
    agent.events, _ = recording_events(console)
    assert agent.run("Посчитай") == "Уточните числа для расчёта."
    assert console.getvalue().splitlines() == [
        "→ Проверяю данные для расчёта…",
        "→ Не удалось выполнить действие: проверьте данные.",
        "→ Рассчитываю 1 ÷ 0…",
        "→ Не удалось посчитать: деление на ноль.",
    ]
    assert "gsk_" not in console.getvalue()
    assert all(marker not in console.getvalue() for marker in TECHNICAL)
