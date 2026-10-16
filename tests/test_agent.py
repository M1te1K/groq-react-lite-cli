import json
import time

import httpx
import pytest

from react_lite.agent import STEP_LIMIT, TIME_LIMIT, TOOL_LIMIT, UNAVAILABLE
from react_lite.llm import LLMUnavailable, Reply, ToolCall


def call(name, args, id="call_1"):
    return Reply(calls=[ToolCall(id, name, json.dumps(args))])


def test_two_dependent_actions_and_context_reset(make_agent):
    def calculate_after_observation(messages):
        assert messages[-1]["role"] == "tool"
        assert messages[-1]["tool_call_id"] == "rate_1"
        rate = json.loads(messages[-1]["content"])["rate_rub_per_unit"]
        return call("calculate", {"operation": "multiply", "a": "500", "b": rate}, "calc_1")

    def summarize(messages):
        calculation = json.loads(messages[-1]["content"])
        rate = json.loads(next(m["content"] for m in messages if m.get("tool_call_id") == "rate_1"))
        assert calculation["result"] == "45061.7000"
        return Reply(
            f"500 USD = {calculation['result']} RUB. Официальный курс ЦБ: "
            f"{rate['rate_rub_per_unit']}, дата {rate['effective_date']}. {rate['source_url']}",
        )

    agent, model = make_agent(
        [
            call("get_exchange_rate", {"currency": "USD", "date": "2026-10-16"}, "rate_1"),
            calculate_after_observation,
            summarize,
            call("calculate", {"operation": "multiply", "a": "1000", "b": "90.1234"}, "calc_2"),
            Reply("1000 USD = 90123.4000 RUB"),
            Reply("Какой курс вас интересует?"),
        ]
    )
    answer = agent.run("Узнай курс USD на сегодня и посчитай стоимость 500 долларов в рублях")
    assert "45 061,70" in answer and "15.10.2026" in answer and "https://www.cbr.ru/" in answer
    assert [m["role"] for m in agent.history] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert "90 123,40" in agent.run("а для 1000 долларов?")
    assert any(m.get("tool_call_id") == "rate_1" for m in model.requests[3])
    agent.reset()
    assert agent.history == []
    agent.run("а для 1000 долларов?")
    assert len(model.requests[-1]) == 2  # Только свежие system и user.


def test_plain_question_can_finish_without_tools(make_agent):
    agent, model = make_agent([Reply("Привет!")])
    assert agent.run("Привет") == "Привет!"
    assert len(model.requests) == 1
    assert all(m["role"] != "tool" for m in agent.history)


def test_model_can_correct_bad_arguments(make_agent):
    def repair(messages):
        assert json.loads(messages[-1]["content"])["error"]["code"] == "invalid_arguments"
        return call("calculate", {"operation": "add", "a": "0.1", "b": "0.2"}, "c2")

    agent, _ = make_agent(
        [
            call("calculate", {"operation": "add", "a": 0.1, "b": "0.2"}),
            repair,
            Reply("0.3"),
        ]
    )
    assert agent.run("Сложи 0.1 и 0.2") == "0.3"


def test_cbr_error_can_be_reported_and_next_request_succeeds(make_agent):
    def report(messages):
        assert json.loads(messages[-1]["content"])["error"]["code"] == "cbr_unavailable"
        return Reply("Не удалось получить официальный курс ЦБ. Попробуйте позже.")

    agent, _ = make_agent(
        [
            call("get_exchange_rate", {"currency": "USD", "date": "2026-10-16"}),
            report,
            Reply("Привет"),
        ],
        handler=lambda r: httpx.Response(503),
    )
    assert "Не удалось" in agent.run("Курс USD?")
    assert agent.run("Привет") == "Привет"


def test_step_limit_leaves_valid_history(make_agent):
    agent, model = make_agent(
        [call("calculate", {"operation": "add", "a": "1", "b": "1"}, f"c{i}") for i in range(3)]
        + [Reply("Работаю")],
        max_steps=3,
    )
    assert agent.run("Считай бесконечно") == STEP_LIMIT
    assert len(model.requests) == 3
    assert agent.history[-2]["role"] == "tool"
    assert agent.run("Ты здесь?") == "Работаю"


def test_multi_call_batch_cannot_bypass_limit(make_agent, monkeypatch):
    reply = Reply(calls=[ToolCall(f"c{i}", "calculate", "{}") for i in range(3)])
    agent, model = make_agent([reply], max_tool_calls=2)
    monkeypatch.setattr(agent.tools, "execute", lambda *a: pytest.fail("Batch must not execute"))
    assert agent.run("Считай") == TOOL_LIMIT
    assert len(model.requests) == 1
    results = [m for m in agent.history if m["role"] == "tool"]
    assert len(results) == 3
    assert {r["tool_call_id"] for r in results} == {"c0", "c1", "c2"}
    assert all(json.loads(r["content"])["error"]["code"] == "tool_limit" for r in results)


def test_tool_limit_across_iterations(make_agent):
    agent, _ = make_agent(
        [call("calculate", {"operation": "add", "a": "1", "b": "1"}, f"c{i}") for i in range(2)],
        max_tool_calls=1,
    )
    assert agent.run("Считай") == TOOL_LIMIT


def test_llm_failure_discards_incomplete_tool_turn(make_agent):
    agent, model = make_agent(
        [
            call("calculate", {"operation": "add", "a": "1", "b": "1"}),
            LLMUnavailable("AuthenticationError", 401),
            Reply("Привет"),
        ]
    )
    assert agent.run("Считай") == UNAVAILABLE
    assert [m["role"] for m in agent.history] == ["user", "assistant"]
    assert agent.run("Привет") == "Привет"
    assert all("tool_calls" not in m for m in model.requests[-1])


def test_hard_deadline_interrupts_blocking_model(make_agent):
    def slow(messages):
        time.sleep(1)
        return Reply("Опоздал")

    agent, _ = make_agent([slow, Reply("Привет")], timeout=0.05)
    start = time.monotonic()
    assert agent.run("Долго") == TIME_LIMIT
    assert time.monotonic() - start < 0.5
    assert agent.run("Привет") == "Привет"


def test_duplicate_tool_call_ids_are_rejected(make_agent):
    duplicate = call("calculate", {"operation": "add", "a": "1", "b": "1"})
    agent, _ = make_agent([duplicate, duplicate])
    assert agent.run("Считай") == UNAVAILABLE
    assert len(agent.history) == 2


def test_no_empty_final_answer(make_agent):
    agent, _ = make_agent([Reply(" ")])
    assert agent.run("Привет") == UNAVAILABLE
