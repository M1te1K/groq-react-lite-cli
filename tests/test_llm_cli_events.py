import io
import json
import logging
import time

import groq
import httpx
import pytest

from react_lite.agent import TIME_LIMIT, UNAVAILABLE
from react_lite.budget import Budget
from react_lite.cli import interact, main
from react_lite.config import Settings
from react_lite.events import Events
from react_lite.llm import GroqModel, LLMUnavailable, Reply


def completion(content="Привет", calls=None, finish="stop"):
    return {
        "id": "chat_1",
        "object": "chat.completion",
        "created": 1,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning": "DO NOT LOG PRIVATE REASONING",
                    "tool_calls": calls,
                },
            }
        ],
    }


def test_sdk_adapter_sends_local_tools_and_preserves_ids():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=completion(
                None,
                [
                    {
                        "id": "chosen_id",
                        "type": "function",
                        "function": {
                            "name": "calculate",
                            "arguments": '{"operation":"add","a":"1","b":"2"}',
                        },
                    }
                ],
                "tool_calls",
            ),
        )

    with groq.Groq(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.Client(
            transport=httpx.MockTransport(handler),
        ),
    ) as client:
        model = GroqModel(client, "openai/gpt-oss-120b")
        reply = model.complete([{"role": "user", "content": "1+2"}], [], Budget(2))
    assert reply.calls[0].id == "chosen_id"
    assert "reasoning" not in reply.as_message()
    assert requests[0]["tool_choice"] == "auto"
    assert requests[0]["parallel_tool_calls"] is False
    assert requests[0]["model"] == "openai/gpt-oss-120b"


def test_cli_agent_and_sdk_round_trip_with_controlled_http(make_agent):
    requests = []

    def tool_reply(name, arguments, call_id):
        return completion(
            None,
            [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
            "tool_calls",
        )

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        messages = body["messages"]
        if len(requests) == 1:
            response = tool_reply(
                "get_exchange_rate",
                {
                    "currency": "USD",
                    "date": "2026-10-16",
                },
                "rate_http",
            )
        elif len(requests) == 2:
            assert messages[-1]["tool_call_id"] == "rate_http"
            rate = json.loads(messages[-1]["content"])["rate_rub_per_unit"]
            assert messages[-2]["tool_calls"][0]["id"] == "rate_http"
            response = tool_reply(
                "calculate",
                {
                    "operation": "multiply",
                    "a": "500",
                    "b": rate,
                },
                "calc_http",
            )
        else:
            assert messages[-1]["tool_call_id"] == "calc_http"
            result = json.loads(messages[-1]["content"])["result"]
            assert result == "45061.7000"
            response = completion(
                f"500 USD = {result} RUB, официальный курс ЦБ 90.1234, "
                "дата 2026-10-15, источник https://www.cbr.ru/"
            )
        return httpx.Response(200, json=response)

    with groq.Groq(
        api_key="test",
        http_client=httpx.Client(
            transport=httpx.MockTransport(handler),
        ),
    ) as client:
        agent, _ = make_agent([])
        agent.model = GroqModel(client, "openai/gpt-oss-120b")
        inputs = iter(
            ["Узнай курс USD на сегодня и посчитай стоимость 500 долларов в рублях", "/exit"]
        )
        output = []
        interact(agent, lambda prompt: next(inputs), output.append)
    assert len(requests) == 3
    assert any("45 061,70" in line and "15.10.2026" in line for line in output)
    assert all("reasoning" not in message for r in requests for message in r["messages"])


@pytest.mark.parametrize(("failure", "attempts"), [(401, 1), (429, 2), (503, 2), ("timeout", 2)])
def test_real_sdk_error_handling_and_retry_count(failure, attempts, monkeypatch, make_agent):
    seen = []
    monkeypatch.setattr(Budget, "backoff", lambda *args: None)

    def handler(request):
        seen.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        return httpx.Response(failure, json={"error": {"message": "gsk_secret"}})

    with groq.Groq(
        api_key="test-key",
        http_client=httpx.Client(
            transport=httpx.MockTransport(handler),
        ),
    ) as client:
        agent, _ = make_agent([])
        agent.model = GroqModel(client, "test", attempts=2)
        assert agent.run("Привет") == UNAVAILABLE
    assert len(seen) == attempts  # SDK defaults не добавляют скрытых повторов.


def test_sdk_retries_obey_global_deadline(make_agent):
    def slow(request):
        time.sleep(1)
        return httpx.Response(503)

    with groq.Groq(
        api_key="test",
        http_client=httpx.Client(
            transport=httpx.MockTransport(slow),
        ),
    ) as client:
        agent, _ = make_agent([], timeout=0.05)
        agent.model = GroqModel(client, "test", attempts=3)
        start = time.monotonic()
        assert agent.run("Привет") == TIME_LIMIT
        assert time.monotonic() - start < 0.5


def test_missing_key_is_unavailable():
    with pytest.raises(LLMUnavailable, match="missing_api_key"):
        GroqModel(None, "test").complete([], [], Budget(1))


@pytest.mark.parametrize("finish", ["length", "content_filter"])
def test_incomplete_generation_not_reported_as_success(finish):
    with groq.Groq(
        api_key="test",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json=completion("partial", finish=finish)),
            )
        ),
    ) as client:
        with pytest.raises(LLMUnavailable, match="incomplete_response"):
            GroqModel(client, "test").complete([], [], Budget(1))


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"choices": [{"finish_reason": "stop", "message": None}]},
        completion(None, [{"id": "call", "type": "function", "function": None}], "tool_calls"),
        completion(None, None, "tool_calls"),
    ],
)
def test_malformed_successful_api_response(body):
    with groq.Groq(
        api_key="test",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json=body),
            )
        ),
    ) as client:
        with pytest.raises(LLMUnavailable):
            GroqModel(client, "test").complete([], [], Budget(1))


def test_logs_and_trace_redact_secrets(make_agent):
    secret = "custom-secret-without-prefix"
    output = io.StringIO()
    logger = logging.getLogger("redaction-test")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(output)
    logger.addHandler(handler)
    trace = []
    agent, _ = make_agent([Reply(f"Ответ {secret} gsk_otherSecret")])
    agent.events = Events(logger, secret, trace.append)
    try:
        agent.run(f"Запрос {secret}")
    finally:
        logger.removeHandler(handler)
    log = output.getvalue()
    assert secret not in log and "gsk_otherSecret" not in log
    assert trace == []  # Итоговое событие больше не дублируется в терминале.
    rows = [json.loads(line) for line in log.splitlines()]
    assert all(r["time"] and r["user_id"] == "local" and r["request_id"] for r in rows)
    assert len({r["request_id"] for r in rows}) == 1
    assert rows[0]["event"] == "request" and rows[-1]["event"] == "response"
    assert all(
        {"Thought", "Action", "Observation"} <= r.keys() for r in rows if r["event"] == "step"
    )


def test_cli_commands_reset_and_continue_after_error(make_agent):
    agent, model = make_agent([LLMUnavailable("timeout"), Reply("Привет")])
    inputs = iter(["", "/help", "/unknown", "Первый", "/reset", "Второй", "/exit"])
    output = []
    interact(agent, lambda prompt: next(inputs), output.append)
    assert any("Неизвестная команда" in line for line in output)
    assert any(UNAVAILABLE in line for line in output)
    assert "Контекст диалога удалён." in output
    assert "Бот: Привет" in output
    assert len(model.requests[-1]) == 2


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, EOFError])
def test_cli_clean_interrupts(make_agent, interrupt):
    agent, _ = make_agent([])

    def read(prompt):
        raise interrupt

    output = []
    interact(agent, read, output.append)
    assert output[-1] == "До свидания!"


def test_cli_settings_do_not_leak_invalid_values(monkeypatch, capsys):
    monkeypatch.setenv("MAX_STEPS", "gsk_secret")
    assert main([]) == 2
    output = capsys.readouterr()
    assert "gsk_secret" not in output.err
    assert "max_steps" in output.err


def test_settings_defaults_and_env_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    (tmp_path / ".env").write_text("GROQ_MODEL=custom-model\nMAX_STEPS=4\n", encoding="utf-8")
    settings = Settings()
    assert settings.groq_model == "custom-model" and settings.max_steps == 4
    assert settings.max_tool_calls == 6
