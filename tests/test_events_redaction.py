import io
import json
import logging
from pathlib import Path
from uuid import uuid4

import groq
import httpx
import pytest

from react_lite.agent import UNAVAILABLE
from react_lite.cli import main
from react_lite.events import Events
from tests.test_llm_cli_events import completion


def string_values(value):
    """Имена полей могут совпадать с секретом; проверяем только значения."""
    if isinstance(value, dict):
        for item in value.values():
            yield from string_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from string_values(item)
    elif isinstance(value, str):
        yield value


@pytest.mark.parametrize(
    "secret",
    [
        "step",
        "Thought",
        "Action",
        "Observation",
        "request_id",
        'quoted"key\\\n',
    ],
)
def test_redacts_nested_values_without_changing_field_names(secret):
    output = io.StringIO()
    logger = logging.getLogger(f"redaction.{uuid4().hex}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.StreamHandler(output)
    logger.addHandler(handler)
    trace = []
    payload = {
        "step": 1,
        "Thought": f"Проверить {secret}",
        "Action": {
            secret: [secret, {"text": "Вложенный ключ gsk_NestedValue"}],
            "path": Path("example") / secret,
            "flag": True,
            "overlap": "gsk_stepNested",
            "gsk_FieldKey": f"Вложенное значение {secret}",
        },
        "Observation": (None, 42, {"value": f"{secret} и gsk_OtherValue"}),
    }
    try:
        Events(logger, secret, trace.append).emit("step", "request-one", **payload)
    finally:
        logger.removeHandler(handler)
        handler.close()
    row = json.loads(output.getvalue())
    assert set(row) == {
        "time",
        "user_id",
        "request_id",
        "event",
        "step",
        "Thought",
        "Action",
        "Observation",
    }
    assert row["step"] == 1
    assert set(row["Action"]) == {secret, "path", "flag", "overlap", "gsk_FieldKey"}
    assert row["Action"][secret] == ["[REDACTED]", {"text": "Вложенный ключ [REDACTED]"}]
    assert row["Action"]["flag"] is True
    assert row["Action"]["overlap"] == "[REDACTED]"
    assert row["Action"]["gsk_FieldKey"] == "Вложенное значение [REDACTED]"
    assert row["Observation"] == [None, 42, {"value": "[REDACTED] и [REDACTED]"}]
    assert payload["Action"][secret][0] == secret  # Исходные данные не изменяются.
    assert trace == []  # Технические step-события остаются только в файле.
    logged_values = list(string_values(row))
    for value in logged_values:
        assert secret not in value
        assert "gsk_" not in value


@pytest.mark.parametrize("secret", ["step", "Thought", "Action", "Observation", "invalid-key"])
def test_main_trace_survives_sdk_401_and_accepts_next_request(
    secret,
    tmp_path,
    monkeypatch,
    capsys,
):
    requests = []

    def handler(request):
        assert request.url.host == "api.groq.com"
        assert request.headers["Authorization"] == f"Bearer {secret}"
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(
                401,
                json={
                    "error": {"message": f"Invalid key {secret} gsk_ErrorValue"},
                },
            )
        return httpx.Response(200, json=completion("Следующий запрос обработан"))

    real_groq = groq.Groq

    def controlled_client(**kwargs):
        return real_groq(
            **kwargs,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

    log_path = tmp_path / "agent.jsonl"
    monkeypatch.chdir(tmp_path)  # Не читаем локальные настройки/ключи проекта.
    monkeypatch.setenv("GROQ_API_KEY", secret)
    monkeypatch.setenv("LOG_FILE", str(log_path))
    monkeypatch.setattr("react_lite.cli.groq.Groq", controlled_client)
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            f"Первый запрос {secret} gsk_UserValue\nВторой запрос\n/exit\n",
        ),
    )
    logger = logging.getLogger("react_lite")
    try:
        assert main(["--trace"]) == 0
    finally:
        for log_handler in logger.handlers[:]:
            log_handler.close()
            logger.removeHandler(log_handler)
    console = capsys.readouterr()
    assert not console.err
    assert f"Бот: {UNAVAILABLE}" in console.out
    assert "Бот: Следующий запрос обработан" in console.out
    assert console.out.rstrip().endswith("До свидания!")
    assert "Thought:" not in console.out and "Action:" not in console.out
    assert "Observation:" not in console.out
    assert console.out.count(UNAVAILABLE) == 1
    assert console.out.count("Следующий запрос обработан") == 1
    assert len(requests) == 2  # 401 не повторяется; второй запрос дошёл до SDK.
    assert requests[0]["messages"][-1]["content"].startswith("Первый запрос")
    assert requests[1]["messages"][-1]["content"] == "Второй запрос"
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len({row["request_id"] for row in rows}) == 2
    assert any(
        row.get("category") == "AuthenticationError" and row.get("status") == 401 for row in rows
    )
    assert [row["answer"] for row in rows if "answer" in row] == [
        UNAVAILABLE,
        "Следующий запрос обработан",
    ]
    assert all({"time", "user_id", "request_id", "event"} <= row.keys() for row in rows)
    steps = [row for row in rows if "Thought" in row]
    assert steps
    assert all({"step", "Thought", "Action", "Observation"} <= row.keys() for row in steps)
    assert rows[0]["user"] == "Первый запрос [REDACTED] [REDACTED]"
    for value in string_values(rows):
        assert secret not in value
        assert "gsk_" not in value
